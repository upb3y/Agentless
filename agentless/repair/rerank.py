import argparse
import json
import logging
import os
from collections import Counter, OrderedDict
from pathlib import Path

from tqdm import tqdm

from agentless.util.postprocess_data import extract_python_blocks, normalize_patch
from agentless.util.utils import load_json, load_jsonl

execution_results = dict()


def _load_results(args):
    global execution_results

    # assumes interval
    interval = (0, args.num_samples - 1)
    root = Path(args.patch_folder)

    for i in range(interval[0], interval[1] + 1):
        patches = load_jsonl(root / f"output_{i}_normalized.jsonl")
        print(
            f"Loaded {len(patches)} patches from {root / f'output_{i}_normalized.jsonl'}"
        )
        for patch in patches[:300]:
            try:
                execution_results.setdefault(patch["instance_id"], []).append(
                    {
                        "normalized_patch": patch["normalized_patch"].strip(),
                        "patch": patch["model_patch"],
                        "plausible": True,  # default to TRUE for now, TODO: add plausible execution.
                    }
                )
            except:
                print(i)
                print(patch)
                exit(-1)


def get_sample(instance_id, sample_id) -> tuple[str, bool]:
    """Returns the diff and pass status."""
    return execution_results[instance_id][sample_id]


def get_all_patches(instance_id, num_samples, deduplicate) -> list[str]:
    """Returns all unique patches."""
    patches = [execution_results[instance_id][i]["patch"] for i in range(num_samples)]
    if deduplicate:
        patch_keys = [
            execution_results[instance_id][i]["normalized_patch"]
            for i in range(num_samples)
        ]
    else:
        patch_keys = [
            execution_results[instance_id][i]["patch"] for i in range(num_samples)
        ]
    unique_patches = set()
    patch_ids = []
    for i in range(num_samples):
        patch_key = patch_keys[i].strip()
        if patch_key and patch_key not in unique_patches:
            unique_patches.add(patch_key)
            patch_ids.append(i)
    return [(id, patches[id]) for id in patch_ids]


def get_all_patches_num(instance_id, num_samples, deduplicate) -> list[str]:
    """Returns all unique patches with number."""
    # print(f"{len(execution_results)}")
    patches = [execution_results[instance_id][i]["patch"] for i in range(num_samples)]
    if deduplicate:
        patch_keys = [
            execution_results[instance_id][i]["normalized_patch"]
            for i in range(num_samples)
        ]
    else:
        patch_keys = [
            execution_results[instance_id][i]["patch"] for i in range(num_samples)
        ]
    unique_patches = {}
    total_patch_num = {}
    patch_ids = []
    for i in range(num_samples):
        if patch_keys[i] and patch_keys[i] not in unique_patches:
            unique_patches[patch_keys[i]] = i
            patch_ids.append(i)
            total_patch_num[i] = 0
        if patch_keys[i]:
            total_patch_num[unique_patches[patch_keys[i]]] += 1

    return [(id, patches[id], total_patch_num[id]) for id in patch_ids]


######

import json


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def majority_voting(args):
    all_pred = []

    for instance_id in execution_results:
        patch_keys = [
            execution_results[instance_id][i]["normalized_patch"]
            for i in range(args.num_samples)
        ]
        plausible = [
            execution_results[instance_id][i]["plausible"]
            for i in range(args.num_samples)
        ]

        raw_patches = [
            execution_results[instance_id][i]["patch"]
            for i in range(args.num_samples)
            for i in range(args.num_samples)
        ]

        if args.plausible:
            patch_ids = list(
                i
                for i in range(args.num_samples)
                if patch_keys[i].strip() and plausible[i]
            )
        else:
            patch_ids = list(
                i for i in range(args.num_samples) if patch_keys[i].strip()
            )

        if not patch_ids:
            # just vote on all patches
            if not all([x.strip() == "" for x in raw_patches]):
                vote = Counter()
                first_appear_idx = dict()
                valid_indices = []
                for i in range(args.num_samples):
                    sample = get_sample(instance_id, i)
                    patch_key = sample["normalized_patch"]
                    if patch_key != "":
                        valid_indices.append(i)
                        vote[patch_key] += 1
                        if patch_key not in first_appear_idx:
                            first_appear_idx[patch_key] = i

                maj_selected_id = max(
                    valid_indices,
                    key=lambda i: (
                        vote[patch_keys[i]],
                        -first_appear_idx[patch_keys[i]],
                    ),
                )
                patch = get_sample(instance_id, maj_selected_id)["patch"]
                all_pred.append(
                    {
                        "model_name_or_path": "agentless",
                        "instance_id": instance_id,
                        "model_patch": patch,
                    }
                )
            else:
                all_pred.append(
                    {
                        "model_name_or_path": "agentless",
                        "instance_id": instance_id,
                        "model_patch": "",
                    }
                )
            continue

        vote = Counter()
        first_appear_idx = dict()
        for i in patch_ids:
            sample = get_sample(instance_id, i)
            patch_key, patch = (
                sample["normalized_patch"],
                sample["patch"],
            )
            vote[patch_key] += 1
            if patch_key not in first_appear_idx:
                first_appear_idx[patch_key] = i
        ### pure majority voting
        maj_selected_id = max(
            patch_ids,
            key=lambda i: (vote[patch_keys[i]], -first_appear_idx[patch_keys[i]]),
        )

        if args.target is not None and instance_id == args.target:
            for patch in vote:
                print(
                    "=" * 20,
                    vote[patch],
                    "=" * 20,
                )
                print(patch)
                print("=" * 50)

        sample = get_sample(instance_id, maj_selected_id)
        all_pred.append(
            {
                "model_name_or_path": "agentless",
                "instance_id": instance_id,
                "model_patch": sample["patch"],
            }
        )

    with open("all_preds.jsonl", "w") as f:
        for pred in all_pred:
            f.write(json.dumps(pred) + "\n")


def normalize_patches(args):
    output_folder = Path(args.patch_folder)
    selected_ids = list(range(args.num_samples))
    for i in selected_ids:
        if os.path.exists(output_folder / f"output_{i}_normalized.jsonl"):
            # skip
            continue
        patches = load_jsonl(output_folder / f"output_{i}_processed.jsonl")
        for d in patches:
            instance_id = d["instance_id"]
            patch = d["model_patch"]
            original_file_content = d["original_file_content"]
            normalized_patch = normalize_patch(
                instance_id, patch, original_file_content
            )
            d["normalized_patch"] = normalized_patch
        with open(output_folder / f"output_{i}_normalized.jsonl", "w") as f:
            for d in patches:
                f.write(json.dumps(d) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_folder", type=str)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=11)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--plausible", action="store_true")
    args = parser.parse_args()

    # first normalize
    normalize_patches(args)
    # then load results
    _load_results(args)
    # then rerank
    majority_voting(args)


if __name__ == "__main__":
    main()
#
