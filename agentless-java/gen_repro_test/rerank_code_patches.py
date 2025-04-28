# rerank_code_patches.py (Revised)
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility function to load JSONL files (same as before)
def load_jsonl(file_path):
    """Loads data from a JSONL file."""
    data = []
    path = Path(file_path)
    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line #{i+1} in {file_path}: {line.strip()} - Error: {e}")
    else:
        logging.error(f"File not found: {file_path}")
    return data

def load_and_process_results(args):
    """
    Loads candidate patches and aggregates their reproduction test outcomes.

    Returns:
        dict: A dictionary where keys are instance_ids and values are lists of
              aggregated results for each candidate code patch for that instance.
              Each aggregated result is a dict like:
              {
                  "code_generation_index": int,
                  "patch": str,              # The candidate code patch string
                  "passed_regression": bool, # Did the regression test pass?
                  "reproduced_count": int    # How many repro tests yielded "REPRODUCED"?
              }
    """
    # --- Load Candidate Code Patches ---
    # Expecting: instance_id, code_generation_index, model_patch
    logging.info(f"Loading candidate code patches from: {args.candidate_patches_file}")
    patches_data = load_jsonl(args.candidate_patches_file)
    candidate_patches = {} # {(instance_id, code_gen_index): patch_string}
    if not patches_data:
        logging.error(f"No candidate patches loaded from {args.candidate_patches_file}. Cannot proceed.")
        return None
    for entry in patches_data:
        if 'instance_id' in entry and 'code_generation_index' in entry and 'model_patch' in entry:
            key = (entry['instance_id'], entry['code_generation_index'])
            candidate_patches[key] = entry['model_patch']
        else:
            logging.warning(f"Skipping candidate patch entry missing required fields (instance_id, code_generation_index, model_patch): {entry}")
    logging.info(f"Loaded {len(candidate_patches)} candidate code patch entries.")
    if not candidate_patches:
        logging.error("No valid candidate patches loaded. Cannot proceed.")
        return None

    # --- Load and Process Reproduction Test Outcomes ---
    # Expecting: instance_id, code_generation_index, generation_index (repro_gen_index),
    #            regression_outcome, repro_test_outcome
    logging.info(f"Loading reproduction test results from: {args.repro_results_file}")
    repro_outcomes_raw = load_jsonl(args.repro_results_file)
    if not repro_outcomes_raw:
        logging.error(f"No reproduction test results loaded from {args.repro_results_file}. Cannot proceed.")
        return None

    # Aggregate results per candidate code patch
    # {(instance_id, code_gen_index): {"passed_regression": bool, "reproduced_count": int}}
    aggregated_results = defaultdict(lambda: {"passed_regression": None, "reproduced_count": 0})

    processed_outcomes = 0
    missing_candidate_keys = set()
    for outcome_entry in repro_outcomes_raw:
        # Check for required fields from the results file
        req_keys = ['instance_id', 'code_generation_index', 'generation_index', 'regression_outcome', 'repro_test_outcome']
        if not all(key in outcome_entry for key in req_keys):
            logging.warning(f"Skipping outcome entry missing required fields ({req_keys}): {outcome_entry}")
            continue

        instance_id = outcome_entry['instance_id']
        code_gen_index = outcome_entry['code_generation_index']
        candidate_key = (instance_id, code_gen_index)

        # Check if this candidate patch exists in our loaded candidates
        if candidate_key not in candidate_patches:
            if candidate_key not in missing_candidate_keys:
                 logging.warning(f"Found results for candidate {candidate_key} but this candidate was not found in {args.candidate_patches_file}. Skipping its results.")
                 missing_candidate_keys.add(candidate_key)
            continue

        # Process regression outcome (only need to check it once per candidate)
        current_agg = aggregated_results[candidate_key]
        if current_agg["passed_regression"] is None: # First time seeing this candidate
            current_agg["passed_regression"] = (outcome_entry['regression_outcome'] == "PASS")
            if not current_agg["passed_regression"]:
                 logging.debug(f"Candidate {candidate_key} did not pass regression ({outcome_entry['regression_outcome']}). Reproduction counts will be ignored unless overridden by another entry.")
                 # If any entry for this candidate fails regression, mark it as failed.
                 # We assume regression outcome is consistent across all repro tests for the same code candidate run.

        # Only count reproduction if regression passed
        if current_agg["passed_regression"] and outcome_entry['repro_test_outcome'] == "REPRODUCED":
            current_agg["reproduced_count"] += 1

        processed_outcomes += 1

    logging.info(f"Processed {processed_outcomes} reproduction test outcomes.")

    # --- Combine aggregated results with patch strings ---
    final_aggregation = defaultdict(list) # {instance_id: [agg_result_dict, ...]}
    for (instance_id, code_gen_index), agg_data in aggregated_results.items():
        # Ensure the candidate patch existed
        if (instance_id, code_gen_index) in candidate_patches:
            final_aggregation[instance_id].append({
                "code_generation_index": code_gen_index,
                "patch": candidate_patches[(instance_id, code_gen_index)],
                "passed_regression": agg_data["passed_regression"],
                "reproduced_count": agg_data["reproduced_count"] if agg_data["passed_regression"] else 0 # Ensure count is 0 if regression failed
            })
        # else: The warning about missing keys was already logged

    logging.info(f"Finished aggregation for {len(final_aggregation)} instances.")
    return dict(final_aggregation) # Convert back to regular dict


def select_best_patch(args, aggregated_results):
    """
    Selects the best patch for each instance based on regression results and
    the count of successful reproductions ('REPRODUCED' outcome).
    """
    final_selection = []
    instances_with_selection = 0
    instances_without_reproducing_patch = 0

    logging.info("Selecting best patch per instance...")
    # Sort instance IDs for deterministic output order
    sorted_instance_ids = sorted(aggregated_results.keys())

    for instance_id in sorted_instance_ids:
        candidate_results = aggregated_results[instance_id]

        # Filter for candidates that passed regression AND reproduced the issue at least once
        eligible_candidates = [
            cand for cand in candidate_results
            if cand["passed_regression"] and cand["reproduced_count"] > 0
        ]

        if not eligible_candidates:
            # Check why no eligible candidates were found
            passed_regression_but_no_repro = any(cand["passed_regression"] and cand["reproduced_count"] == 0 for cand in candidate_results)
            failed_regression = any(not cand["passed_regression"] for cand in candidate_results)

            status = "No eligible patches found"
            if passed_regression_but_no_repro and not failed_regression:
                status = "No patches reproduced the issue (though some passed regression)"
            elif failed_regression and not passed_regression_but_no_repro:
                 status = "All candidate patches failed regression tests"
            elif failed_regression and passed_regression_but_no_repro:
                 status = "Some patches failed regression, others passed but did not reproduce the issue"

            logging.warning(f"Instance {instance_id}: {status}. No patch selected.")
            instances_without_reproducing_patch += 1
            final_selection.append({
                "model_name_or_path": args.model_name,
                "instance_id": instance_id,
                "model_patch": "", # Empty patch indicates failure to select
                "status": status,
                "selected_code_generation_index": None,
                "selected_reproduced_count": 0
            })
            continue

        # --- Select the best among eligible candidates ---
        # Sort by:
        # 1. Highest reproduced_count (descending)
        # 2. Lowest code_generation_index (ascending - for tie-breaking)
        eligible_candidates.sort(key=lambda x: (-x["reproduced_count"], x["code_generation_index"]))

        winner = eligible_candidates[0] # The best candidate after sorting

        instances_with_selection += 1
        logging.info(f"Instance {instance_id}: Selected winning patch (CodeGenIndex: {winner['code_generation_index']}, Reproduced Count: {winner['reproduced_count']})")

        result = {
            "model_name_or_path": args.model_name,
            "instance_id": instance_id,
            "model_patch": winner["patch"], # The best raw patch string
            "status": "Selected by max reproductions (passed regression)",
            "selected_code_generation_index": winner['code_generation_index'],
            "selected_reproduced_count": winner['reproduced_count']
        }
        final_selection.append(result)

    logging.info(f"Selection complete. Selected patches for {instances_with_selection} instances.")
    logging.warning(f"Could not select a reproducing patch (that passed regression) for {instances_without_reproducing_patch} instances.")
    return final_selection

def main():
    parser = argparse.ArgumentParser(description="Rerank candidate code patches based on reproduction test success counts.")

    # Input Arguments
    parser.add_argument(
        "--candidate_patches_file",
        type=str,
        required=True,
        help="Path to JSONL file containing multiple candidate code patches per instance. Requires: instance_id, code_generation_index, model_patch."
    )
    parser.add_argument(
        "--repro_results_file",
        type=str,
        required=True,
        help="Path to the combined JSONL output from run_java_tests.py runs. Requires: instance_id, code_generation_index, generation_index (repro_gen_index), regression_outcome, repro_test_outcome."
    )

    # Output Argument
    parser.add_argument(
        "--output_file",
        type=str,
        default="reranked_patches.jsonl",
        help="Path to save the selected best patch for each instance."
    )

    # Configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="candidate_model",
        help="Identifier for the model/process that generated the candidate patches (used in output)."
    )

    args = parser.parse_args()

    # 1. Load candidates and aggregate test outcomes
    aggregated_results = load_and_process_results(args)

    if aggregated_results is None:
        logging.error("Failed to load or process execution results. Exiting.")
        return # Exit if loading/processing failed

    # 2. Select the best patch per instance based on counts and tie-breaking
    final_patches = select_best_patch(args, aggregated_results)

    # 3. Write the final selected patches
    logging.info(f"Saving {len(final_patches)} selected patch entries to {args.output_file}")
    try:
        with open(args.output_file, "w", encoding='utf-8') as f:
            for entry in final_patches:
                f.write(json.dumps(entry) + "\n")
        logging.info("Successfully saved final patches.")
    except IOError as e:
        logging.error(f"Failed to write output file {args.output_file}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during writing: {e}", exc_info=True)

    logging.info("Script finished.")

if __name__ == "__main__":
    # Need defaultdict
    from collections import defaultdict
    # Need Path from pathlib
    from pathlib import Path
    # Basic logging config
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()