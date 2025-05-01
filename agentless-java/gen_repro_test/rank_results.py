import json
import os
import logging
import argparse
from collections import defaultdict

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_jsonl(file_path):
    """Loads data from a JSONL file."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {line.strip()} - Error: {e}")
    else:
        logging.error(f"File not found: {file_path}")
    return data

def rank_and_select_patches(results_filepath, patches_filepath, output_filepath):
    """
    Ranks patches based on reproduction test outcomes and selects the best one per instance.

    Args:
        results_filepath: Path to the test run results JSONL file.
        patches_filepath: Path to the (cleaned) patches JSONL file containing patch content.
        output_filepath: Path to save the ranked selection results.
    """
    logging.info("Loading test run results...")
    run_results = load_jsonl(results_filepath)
    if not run_results:
        logging.error("Failed to load run results or file is empty. Exiting.")
        return

    logging.info("Loading patch data...")
    all_patches_data = load_jsonl(patches_filepath)
    if not all_patches_data:
        logging.error("Failed to load patch data or file is empty. Exiting.")
        return

    # Create a lookup dictionary for patch content: key = (instance_id, repair_patch_index)
    patch_content_lookup = {}
    for patch_item in all_patches_data:
        try:
            instance_id = patch_item.get("instance_id")
            # Use generation_index if present, otherwise original index from load_jsonl needed
            # Let's assume repair_patch_index was correctly added during cleaning or is present
            repair_index = patch_item.get("repair_patch_index", patch_item.get("generation_index"))
            model_patch = patch_item.get("model_patch")
            model_name = patch_item.get("model_name_or_path")

            if instance_id is not None and repair_index is not None and model_patch is not None:
                 key = (instance_id, repair_index)
                 if key in patch_content_lookup:
                      logging.warning(f"Duplicate patch key found: {key}. Overwriting with later entry.")
                 patch_content_lookup[key] = {"model_patch": model_patch, "model_name_or_path": model_name}
            else:
                 logging.warning(f"Skipping patch data item due to missing keys: {patch_item}")

        except Exception as e:
            logging.error(f"Error processing patch data item: {patch_item}. Error: {e}")

    logging.info(f"Created lookup for {len(patch_content_lookup)} patches.")

    # Group results by instance_id
    results_by_instance = defaultdict(list)
    for result in run_results:
        instance_id = result.get("instance_id")
        if instance_id:
            results_by_instance[instance_id].append(result)
        else:
            logging.warning(f"Skipping result item due to missing 'instance_id': {result}")

    logging.info(f"Grouped results for {len(results_by_instance)} unique instance IDs.")

    final_selection = []
    selected_count = 0
    not_selected_count = 0

    # Rank and select within each group
    for instance_id, results in results_by_instance.items():
        reproducing_patches = []
        for res in results:
            outcome = res.get("outcome")
            # Consider only successful reproductions
            if outcome == "REPRODUCED":
                repair_index = res.get("repair_patch_index")
                if repair_index is not None:
                    reproducing_patches.append({
                        "repair_patch_index": repair_index,
                        "outcome": outcome,
                        # Include run_id for reference if needed later
                        "run_id": res.get("run_id")
                    })
                else:
                    logging.warning(f"Found 'REPRODUCED' outcome for {instance_id} but missing 'repair_patch_index'. Skipping.")

        selected_patch_info = None
        if reproducing_patches:
            # Sort by repair_patch_index to pick the lowest one consistently
            reproducing_patches.sort(key=lambda x: x["repair_patch_index"])
            selected_patch_info = reproducing_patches[0]
            selected_count += 1
            logging.info(f"Selected patch index {selected_patch_info['repair_patch_index']} for instance {instance_id} (Outcome: REPRODUCED).")
        else:
            # No patch successfully reproduced the issue
            not_selected_count += 1
            logging.info(f"No patch reproduced the issue for instance {instance_id}.")

        # Prepare output entry
        output_entry = {"instance_id": instance_id}
        if selected_patch_info:
            selected_key = (instance_id, selected_patch_info["repair_patch_index"])
            patch_details = patch_content_lookup.get(selected_key)

            if patch_details:
                output_entry["selected_repair_patch_index"] = selected_patch_info["repair_patch_index"]
                output_entry["selected_model_name_or_path"] = patch_details.get("model_name_or_path", "N/A")
                output_entry["selected_model_patch"] = patch_details.get("model_patch")
                output_entry["reproduction_outcome"] = "REPRODUCED"
            else:
                logging.error(f"Could not find patch details in lookup for selected key: {selected_key}. Selection failed for {instance_id}")
                # Mark as failed selection despite finding a reproducing outcome
                output_entry["selected_repair_patch_index"] = selected_patch_info["repair_patch_index"] # Keep index for debugging
                output_entry["selected_model_name_or_path"] = None
                output_entry["selected_model_patch"] = None
                output_entry["reproduction_outcome"] = "SELECTION_ERROR_PATCH_LOOKUP_FAILED"
                # Decrement selected count as it ultimately failed
                selected_count -=1
                not_selected_count +=1

        else:
            output_entry["selected_repair_patch_index"] = None
            output_entry["selected_model_name_or_path"] = None
            output_entry["selected_model_patch"] = None
            output_entry["reproduction_outcome"] = "NO_REPRODUCTION"

        final_selection.append(output_entry)

    logging.info(f"Ranking complete. Selected patches for {selected_count} instances. No reproducing patch found for {not_selected_count} instances.")

    # Write output file
    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        # Sort final output by instance_id for consistency
        final_selection.sort(key=lambda x: x['instance_id'])

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for entry in final_selection:
                outfile.write(json.dumps(entry) + '\n')
        logging.info(f"Successfully wrote selection results to {output_filepath}")

    except IOError as e:
        logging.error(f"Error writing output file {output_filepath}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during output writing: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank and select reproduction patches based on test outcomes.")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to the JSONL file containing test run outcomes (e.g., test_run_results.jsonl).")
    parser.add_argument("--patches_file", type=str, required=True,
                        help="Path to the JSONL file containing the patch content and metadata (e.g., filtered_patches_cleaned.jsonl).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the final ranked/selected patches (JSONL format).")

    args = parser.parse_args()

    rank_and_select_patches(args.results_file, args.patches_file, args.output_file)

    print("\nRanking script finished. Check logs and the output file.")