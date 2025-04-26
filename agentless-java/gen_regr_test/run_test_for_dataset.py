# run_test_for_dataset.py (CLI Version)

import argparse
import subprocess
import os
import time
import traceback
import logging
import sys # To get python executable path

# --- Import dataset loading library ---
try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Please install it using: pip install datasets")
    exit(1)

# --- Import jsonlines for checking output existence (optional) ---
# Not strictly needed for writing, as the subprocess handles it
try:
    import jsonlines
except ImportError:
    print("Error: 'jsonlines' library not found. Please install it using: pip install jsonlines")
    exit(1)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
    ]
)
log = logging.getLogger(__name__)

# --- Path to the script to be called ---
# Assumes find_pass_java_test.py is in the same directory
FIND_PASS_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "find_passing_java_tests.py")

# --- Main Processing Function ---
def process_dataset_entry_via_cli(
    entry: dict,
    output_file: str,
    timeout: int,
    run_id: str
) -> bool:
    """
    Processes a single dataset entry by calling find_pass_java_test.py
    via the command line. Returns True on success (exit code 0), False otherwise.
    """
    repo_short_name = entry.get("repo")
    commit_hash = entry.get("base_commit")
    instance_id = entry.get("instance_id") # Primarily for logging here

    if not repo_short_name or not commit_hash or not instance_id:
        log.error(f"Skipping entry due to missing repo, base_commit, or instance_id: {entry}")
        return False

    # Construct full GitHub URL (adjust if source is different)
    repo_url = f"https://github.com/{repo_short_name}.git"
    log.info(f"\n--- Processing Instance via CLI: {instance_id} ---")
    log.info(f"Repo URL: {repo_url}")
    log.info(f"Commit Hash: {commit_hash}")
    
    # Construct the command line arguments
    command = [
        sys.executable, # Use the same python interpreter that's running this script
        FIND_PASS_SCRIPT_PATH,
        "--repo_url", repo_url,
        "--commit_hash", commit_hash,
        "--output_file", output_file, # Pass the final output file path
        "--instance_id", instance_id
    ]

    log.info(f"Executing command: {' '.join(command)}")

    try:
        # Execute the command
        # We don't need the timeout here if the subprocess handles its own Docker timeout
        process = subprocess.run(
            command,
            capture_output=False, # Capture stdout/stderr
            text=True,           # Decode output as text
            check=False,         # Don't raise exception on non-zero exit
            encoding='utf-8',
            errors='replace'
        )

        # Log output from the subprocess
        if process.stdout:
            log.info(f"--- Subprocess STDOUT [{instance_id}] ---")
            log.info(process.stdout)
            log.info(f"--- End Subprocess STDOUT [{instance_id}] ---")
        if process.stderr:
            # Log stderr as warning or error depending on return code
            if process.returncode == 0:
                log.warning(f"--- Subprocess STDERR (Warning) [{instance_id}] ---")
                log.warning(process.stderr)
                log.warning(f"--- End Subprocess STDERR [{instance_id}] ---")
            else:
                log.error(f"--- Subprocess STDERR (Error) [{instance_id}] ---")
                log.error(process.stderr)
                log.error(f"--- End Subprocess STDERR [{instance_id}] ---")

        # Check the return code
        if process.returncode == 0:
            log.info(f"Subprocess for {instance_id} completed successfully.")
            # The subprocess should have appended its result to output_file
            return True
        else:
            log.error(f"Subprocess for {instance_id} failed with return code {process.returncode}.")
            # Even on failure, the subprocess might have written *something* (like an error state)
            # to the output file, or it might have failed before writing anything.
            return False

    except FileNotFoundError:
        log.error(f"Error: Could not find Python executable '{sys.executable}' or script '{FIND_PASS_SCRIPT_PATH}'.")
        log.error(traceback.format_exc())
        return False
    except Exception as e:
        log.error(f"An unexpected error occurred while running subprocess for {instance_id}: {e}")
        log.error(traceback.format_exc())
        return False


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run find_pass_java_test (via CLI) on multiple repositories from a Hugging Face dataset."
    )
    parser.add_argument("--dataset_name", default="Daoguang/Multi-SWE-bench", help="Name of the Hugging Face dataset.")
    parser.add_argument("--dataset_split", default="java_verified", help="Dataset split to process (e.g., 'java_verified', 'train', 'test').")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file where results will be appended.")
    parser.add_argument("--start_index", type=int, default=0, help="Index of the dataset entry to start processing from (inclusive).")
    parser.add_argument("--end_index", type=int, default=None, help="Index of the dataset entry to stop processing at (exclusive). Default processes till the end.")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds PASSED to find_pass_java_test.py for Docker execution per repository.")
    parser.add_argument("--run_id", default=f"dataset_run_cli_{int(time.time())}", help="Unique ID for this dataset processing run (used for logging).")
    parser.add_argument("--log_file", default=None, help="Optional path to a file for logging output.")

    args = parser.parse_args()

    # --- Configure File Logging if specified ---
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode='a') # Append mode
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        log.addHandler(file_handler)
        log.info(f"Logging also directed to file: {args.log_file}")

    log.info("--- Starting Dataset Java Test Finder (CLI Execution) ---")
    log.info(f"Dataset: {args.dataset_name}, Split: {args.dataset_split}")
    log.info(f"Output File: {args.output_file}")
    log.info(f"Processing Indices: {args.start_index} to {args.end_index if args.end_index is not None else 'end'}")
    log.info(f"Docker Timeout (passed to subprocess): {args.timeout}s")
    log.info(f"Run ID: {args.run_id}")
    # LLM usage is now controlled within the find_pass_java_test.py execution

    # --- Pre-checks ---
    # We assume find_pass_java_test.py does its own docker/git checks
    if not os.path.exists(FIND_PASS_SCRIPT_PATH):
         log.error(f"Error: The script to call ({FIND_PASS_SCRIPT_PATH}) does not exist.")
         exit(1)
    # Docker daemon connection check happens within the called script

    # --- Load Dataset ---
    try:
        log.info(f"Loading dataset {args.dataset_name}, split {args.dataset_split}...")
        dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
        log.info(f"Dataset loaded successfully. Total entries in split: {len(dataset)}")
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        log.error(traceback.format_exc())
        exit(1)

    # --- Determine Processing Range ---
    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else len(dataset)
    end_index = min(end_index, len(dataset)) # Ensure end_index doesn't exceed dataset size

    if start_index >= end_index:
        log.info("Start index is greater than or equal to end index. No entries to process.")
        exit(0)

    log.info(f"Will process entries from index {start_index} up to (but not including) {end_index}.")

    # --- Iterate and Process ---
    processed_count = 0
    success_count = 0
    error_count = 0 # Counts entries where the subprocess failed
    critical_error_count = 0 # Counts errors in this script's loop

    for i in range(start_index, end_index):
        log.info(f"--- Processing dataset index: {i} ---")
        try:
            entry = dataset[i]
            success = process_dataset_entry_via_cli(
                entry=entry,
                output_file=args.output_file,
                timeout=args.timeout,
                run_id=args.run_id
            )
            processed_count += 1
            if success:
                success_count +=1
            else:
                error_count += 1

        except Exception as e:
            # This catches errors in the main loop itself, not subprocess errors
            log.error(f"Critical unhandled error in main loop for index {i}: {e}")
            log.error(traceback.format_exc())
            critical_error_count += 1
            # Attempt to log minimal failure info if possible
            try:
                 minimal_info = {"index": i, "instance_id": dataset[i].get("instance_id", "UNKNOWN"), "status": "error_unhandled_main_loop", "error_message": str(e)}
                 with jsonlines.open(args.output_file, mode="a") as writer:
                     writer.write(minimal_info)
            except Exception as write_e:
                 log.error(f"FATAL: Could not write minimal error info for index {i} to output file: {write_e}")


        log.info(f"--- Finished processing dataset index: {i} ---")
        time.sleep(0.5) # Slightly smaller delay might be okay now

    log.info("--- Dataset Processing Finished ---")
    log.info(f"Attempted to process {processed_count} entries (from index {start_index} to {end_index}).")
    log.info(f"Subprocess executions successful (exit code 0): {success_count}")
    log.info(f"Subprocess executions failed (non-zero exit code): {error_count}")
    log.info(f"Encountered {critical_error_count} critical errors during main loop iteration.")
    log.info(f"Results appended to: {args.output_file}")
    log.info(f"Detailed logs for each run (if any) saved by the subprocess in logs/{args.run_id}/INSTANCE_ID/")
    if args.log_file:
        log.info(f"Overall logs for this runner script saved to: {args.log_file}")