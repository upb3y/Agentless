import os
import subprocess
import re
import tempfile
import shutil
import time
import traceback
import json
import logging
import glob
from typing import List, Optional, Tuple, Dict, Set

# --- Configure Logging ---
# Add specific configuration if running standalone
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(funcName)s] %(message)s' # Added function name
)
log = logging.getLogger(__name__)

# --- Helper Function: clone_and_checkout ---
# (Adapted from logic previously discussed/implied for run_test_for_dataset.py)
def clone_and_checkout(repo_url: str, commit_hash: str, target_dir: str) -> bool:
    """Clones a repo and checks out a specific commit."""
    log.info(f"Cloning {repo_url} to {target_dir}...")
    # Use --depth 1 for faster initial clone, then unshallow if needed
    clone_command = ["git", "clone", "--quiet", "--depth", "1", repo_url, target_dir]
    clone_process = subprocess.run(clone_command, capture_output=True, text=True, check=False)
    if clone_process.returncode != 0:
        # If shallow clone fails, try a full clone
        log.warning(f"Shallow clone failed, attempting full clone: {clone_process.stderr}")
        clone_command = ["git", "clone", "--quiet", repo_url, target_dir]
        clone_process = subprocess.run(clone_command, capture_output=True, text=True, check=False)
        if clone_process.returncode != 0:
            log.error(f"Error cloning repository {repo_url}: {clone_process.stderr}")
            return False
    log.info("Clone successful.")

    log.info(f"Checking out commit {commit_hash}...")
    # Check if commit exists locally first (might be head of shallow clone)
    check_commit_command = ["git", "cat-file", "-e", commit_hash]
    check_commit_process = subprocess.run(check_commit_command, cwd=target_dir, capture_output=True, text=True, check=False)

    if check_commit_process.returncode != 0:
        log.info(f"Commit {commit_hash} not found locally, attempting fetch...")
        # Try unshallowing first if it was a shallow clone
        unshallow_command = ["git", "fetch", "--unshallow", "--quiet"]
        fetch_process = subprocess.run(unshallow_command, cwd=target_dir, check=False, capture_output=True)
        # Always try fetching the specific commit or --all as fallback
        fetch_command = ["git", "fetch", "--quiet", "origin", commit_hash]
        fetch_process_commit = subprocess.run(fetch_command, cwd=target_dir, check=False, capture_output=True)
        if fetch_process_commit.returncode != 0:
             log.warning(f"Could not fetch specific commit {commit_hash}, fetching --all as fallback...")
             fetch_all_command = ["git", "fetch", "--all", "--quiet"]
             subprocess.run(fetch_all_command, cwd=target_dir, check=False)


    # Now attempt checkout
    checkout_command = ["git", "checkout", "--force", commit_hash]
    checkout_process = subprocess.run(checkout_command, cwd=target_dir, capture_output=True, text=True, check=False)
    if checkout_process.returncode != 0:
        log.error(f"Error checking out commit {commit_hash} after fetch attempts: {checkout_process.stderr}")
        return False

    log.info("Checkout successful.")
    # Verify checkout
    verify_command = ["git", "rev-parse", "HEAD"]
    verify_process = subprocess.run(verify_command, cwd=target_dir, capture_output=True, text=True, check=True)
    if not verify_process.stdout.strip().startswith(commit_hash):
         log.warning(f"Checked out commit {verify_process.stdout.strip()} does not match requested {commit_hash}")
         # Decide if this is fatal? For testing a specific base state, it likely is.
         return False

    log.info(f"Verified checkout is at {commit_hash[:12]}")
    return True


# --- Helper Function: parse_maven_surefire_stdout ---
# (Copied from find_pass_java_test.py)
def parse_maven_surefire_stdout(log_data: str) -> List[str]:
    """Parses Maven Surefire stdout logs to find passing tests (class level)."""
    passing_tests: Set[str] = set()
    running_tests: Set[str] = set()
    failed_tests: Set[str] = set()
    error_tests: Set[str] = set()
    # Regex to find lines like "[INFO] Running com.google.gson.functional.DefaultTypeAdaptersTest"
    running_test_class_regex = re.compile(r"\[INFO\] Running ([\w\.$]+)")
    # Regex to find lines indicating failures/errors summarized by Surefire/Failsafe
    # Example: [ERROR] Failures:
    # Example: [ERROR]   ClassName.testMethod:123 Expected X but was Y
    # Example: [ERROR] Errors:
    # Example: [ERROR]   ClassName.initializationError:12 » NullPointer
    # We capture the class name from the lines *following* Failures/Errors headers
    failure_error_header_regex = re.compile(r"\[ERROR\] (?:Failures|Errors):")
    failure_error_detail_regex = re.compile(r"\[ERROR\]\s+([\w\.$]+)(?:\.[\w\$]+)?(?:[:\(]|$)") # Captures class name from test method or direct class error

    # Regex for tests listed under the RESULTS section's Failures/Errors
    # Example: Failures:
    # Example:   SampleTest.testFailure1:50 Assertion Error...
    # Example: Errors:
    # Example:   SampleTest.testError1:60 » RuntimeException
    results_section_fail_err_regex = re.compile(r"^(?:Failures|Errors):\s*$", re.MULTILINE)
    results_detail_fail_err_regex = re.compile(r"^\s+([\w\.$]+)\.[\w\$]+(?::\d+)?(?:$|\s|»)") # Captures class name

    # Regex for summary line
    summary_regex = re.compile(
        r"\[INFO\] Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)"
    )
    overall_success = False
    capture_failures_errors = False # Flag to indicate we are below a Failures/Errors header
    in_results_fail_err_section = False # Flag for RESULTS section

    log_lines = log_data.splitlines()
    for i, line in enumerate(log_lines):
        # Find classes being run
        match = running_test_class_regex.search(line)
        if match:
            running_tests.add(match.group(1))
            continue # Continue to next line

        # Detect start of specific [ERROR] Failures/Errors blocks
        if failure_error_header_regex.search(line):
             capture_failures_errors = True
             in_results_fail_err_section = False # Reset this flag
             continue
        # Capture failures/errors listed directly below the header
        if capture_failures_errors:
            fe_match = failure_error_detail_regex.search(line)
            if fe_match:
                 failed_tests.add(fe_match.group(1)) # Add class name
            # Stop capturing if we hit a blank line or another section
            if not line.strip() or line.startswith("[INFO]") or line.startswith("[WARNING]"):
                 capture_failures_errors = False

        # Detect start of RESULTS section Failures/Errors
        if results_section_fail_err_regex.search(line):
             in_results_fail_err_section = True
             capture_failures_errors = False # Reset this flag
             continue
        # Capture failures/errors within the RESULTS section
        if in_results_fail_err_section:
             res_match = results_detail_fail_err_regex.search(line)
             if res_match:
                  failed_tests.add(res_match.group(1)) # Add class name
             # Stop capturing if indentation changes or different section starts
             if not line.startswith("  ") and line.strip(): # Simple check
                  in_results_fail_err_section = False


        # Check overall summary line
        match = summary_regex.search(line)
        if match:
            tests_run_count = int(match.group(1))
            failures_count = int(match.group(2))
            errors_count = int(match.group(3))
            # Consider overall success only if the final build summary says so
            # This regex check happens later on the whole log_data

    if re.search(r"\[INFO\] BUILD SUCCESS", log_data):
        overall_success = True

    # Combine failed and error tests (as we capture class names)
    failed_or_error_tests = failed_tests.union(error_tests)

    # Deduce passing tests: start with all run, remove known failed/errored
    passing_tests = running_tests - failed_or_error_tests

    # Handle edge cases
    if overall_success and not passing_tests and running_tests:
        log.warning("Maven BUILD SUCCESS reported, but no specific passing tests identified from 'Running...' lines minus failures/errors. Assuming all run tests passed (check logs carefully).")
        passing_tests = running_tests
    elif not overall_success and not failed_or_error_tests and running_tests:
        log.warning("Maven BUILD FAILURE/ERROR reported, but no specific failed tests identified. Cannot determine passing tests reliably from stdout.")
        # In case of build failure with no specific test failures identified,
        # it's safer to report no passing tests.
        passing_tests = set()
    elif not running_tests and (failed_or_error_tests or not overall_success):
         log.warning("No 'Running test...' lines found. Cannot determine passing tests.")
         passing_tests = set()


    log.info(f"Maven stdout parsing: Found {len(running_tests)} running classes, {len(failed_or_error_tests)} failed/errored classes. Deduced {len(passing_tests)} passing classes.")
    return sorted(list(passing_tests))

# --- Helper Function: parse_gradle_stdout ---
# (Copied from find_pass_java_test.py)
def parse_gradle_stdout(log_data: str) -> List[str]:
    """Parses Gradle stdout logs to find passing test methods."""
    # Uses fully qualified method names: e.g., com.package.ClassName.testMethod
    passing_tests: Set[str] = set()
    failed_tests: Set[str] = set()
    tests_executed = False
    build_successful = False
    # Regex to find lines indicating a test method passed/failed/skipped
    # Example: com.example.MyTestClass > myMethod() PASSED
    # Example: com.example.sub.MyOtherTest > anotherTest FAILED
    # Example: com.example.sub.MyOtherTest > skippedTest SKIPPED
    # Captures Class + Method + Result
    # Updated to handle potential parametrization like method[1]
    test_result_regex = re.compile(r"^([\w\.\$]+)\s+>\s+([\w\$]+(?:\[[^\]]+\])?)\(?\)?\s+(PASSED|FAILED|SKIPPED)\s*$", re.MULTILINE)

    # Regex for summary line (less reliable for individual tests but good for context)
    # Example: 10 tests completed, 1 failed, 2 skipped
    summary_regex = re.compile(r"(\d+) tests completed, (\d+) failed, (\d+) skipped")

    build_success_regex = re.compile(r"^BUILD SUCCESSFUL", re.MULTILINE)
    build_failed_regex = re.compile(r"^BUILD FAILED", re.MULTILINE)
    test_task_regex = re.compile(r"> Task :(\w+:)*test\b") # Matches :test, :subproject:test etc.

    if test_task_regex.search(log_data): tests_executed = True
    if build_success_regex.search(log_data): build_successful = True

    for match in test_result_regex.finditer(log_data):
        class_name = match.group(1)
        method_name = match.group(2)
        result = match.group(3)

        # Construct a unique identifier, handle potential inner classes ($)
        # Use class + method name
        test_id = f"{class_name}.{method_name}"

        if result == "PASSED":
            passing_tests.add(test_id)
        elif result == "FAILED":
            failed_tests.add(test_id)
        # We ignore SKIPPED for regression check purposes

    # Refine passing tests: Remove any that were somehow marked both passed and failed
    # (unlikely in standard Gradle output, but a safeguard)
    passing_tests = passing_tests - failed_tests

    # Contextual Warnings
    if tests_executed and not build_successful:
        # Check if build failed *before* tests could complete
        if not build_failed_regex.search(log_data):
             log.warning("Gradle tests executed, but BUILD SUCCESSFUL message not found. Test results might be incomplete.")
        else: # Build definitely failed
             log.warning("Gradle build failed. Passing tests identified before failure might be incomplete or misleading.")
             # If the build failed hard, perhaps no tests reliably passed. Consider clearing passing_tests?
             # For now, we report what was parsed as PASSED before the failure.

    if build_successful and tests_executed and not passing_tests and not failed_tests:
        log.warning("Gradle build successful and tests ran (:test task executed), but no specific PASSED/FAILED test method lines identified via regex. Test output format might differ, or no tests were found/run.")

    log.info(f"Gradle stdout parsing: Found {len(passing_tests)} passed methods, {len(failed_tests)} failed methods.")
    return sorted(list(passing_tests))


# --- Helper Function: run_tests_in_docker ---
# (Copied from find_pass_java_test.py and adapted slightly)
def run_tests_in_docker(
    local_repo_path: str,
    commit_hash: str, # Base commit hash for context
    test_command: str,
    instance_id: str, # Unique ID for this specific test run (e.g., "proj_abc123_patch_test")
    docker_image: str,
    timeout: int = 1800,
    run_id: str = "patch_test_run" # Base directory for logs for this type of run
) -> List[str]:
    """Uses Docker to run tests from a locally mounted repository."""
    passing_tests: List[str] = []
    start_time = time.time()
    container_repo_path = "/usr/src/repo" # Standard mount point inside container

    # Create a unique log directory for this specific instance run
    log_dir = os.path.join("logs", run_id, instance_id)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        log.error(f"Failed to create log directory {log_dir}: {e}. Logs will not be saved.")
        # Decide if this is fatal? Probably should continue execution if possible.

    absolute_repo_path = os.path.abspath(local_repo_path)
    if not os.path.isdir(absolute_repo_path):
        log.error(f"Error: Local repository path does not exist or is not a directory: {absolute_repo_path}")
        return [] # Cannot proceed

    # Ensure gradlew is executable if present (needed for Gradle projects)
    gradlew_path = os.path.join(absolute_repo_path, "gradlew")
    if os.path.isfile(gradlew_path):
        try:
            os.chmod(gradlew_path, os.stat(gradlew_path).st_mode | 0o111)
        except OSError as e:
            log.warning(f"Could not make gradlew executable: {e}")


    # Construct Docker command
    # Mount repo read-write in case build process generates files
    # Use --workdir to ensure commands run in the repo root
    # Use /bin/bash -c to execute the potentially complex test command
    # Add --rm to clean up container afterwards
    docker_run_command = [
        "docker", "run", "--rm",
        "-v", f"{absolute_repo_path}:{container_repo_path}",
        "--workdir", container_repo_path,
        docker_image,
        "/bin/bash", "-c",
        # Added echo markers for easier log debugging
        f"echo '--- Docker Container Start ---' && echo 'Running command:' && echo '{test_command}' && {test_command} && echo '--- Docker Container End ---'"
    ]

    log.info(f"--- Running Docker for instance: {instance_id} ---")
    log.info(f"Host Path: {absolute_repo_path}")
    log.info(f"Container Path: {container_repo_path}")
    log.info(f"Docker Image: {docker_image}")
    log.info(f"Timeout: {timeout}s")
    log.info(f"Executing command: {test_command}") # Log the core command

    log_file_path = os.path.join(log_dir, "test_execution.log")
    log_data = f"Instance ID: {instance_id}\nBase Commit Hash: {commit_hash}\n"
    log_data += f"Docker Image: {docker_image}\nTest Command: {test_command}\nTimeout: {timeout}\n"

    try:
        process = subprocess.run(
            docker_run_command, capture_output=True, text=True, timeout=timeout,
            check=False, encoding='utf-8', errors='replace'
        )
        # Log execution details
        log_data += f"Return Code: {process.returncode}\n"
        log_data += f"\n--- STDOUT ---\n{process.stdout}"
        log_data += f"\n--- STDERR ---\n{process.stderr}"
        success = process.returncode == 0 # Basic success check

        log.info(f"Docker execution finished for {instance_id} with return code {process.returncode}.")

        # Parse results based on the command used
        # It's crucial that the parsers can handle logs even if the build didn't fully succeed
        # (e.g., if tests ran but the overall 'mvn install' failed later)
        if "mvn" in test_command.lower():
            log.info(f"Parsing Maven output for {instance_id}...")
            passing_tests = parse_maven_surefire_stdout(log_data)
        elif "gradle" in test_command.lower():
            log.info(f"Parsing Gradle output for {instance_id}...")
            passing_tests = parse_gradle_stdout(log_data)
        else:
            log.warning(f"Unknown test command type in '{test_command}'. Cannot automatically parse results.")
            passing_tests = [] # Cannot determine passing tests

        log.info(f"Identified {len(passing_tests)} passing tests for {instance_id}.")

    except subprocess.TimeoutExpired:
        log.error(f"Error: Docker execution timed out for {instance_id} after {timeout} seconds.")
        log_data += "\n--- RESULT: TIMEOUT ---"
        passing_tests = [] # Timeout means tests didn't complete successfully
    except Exception as e:
        log.error(f"Error: An unexpected error occurred during Docker execution for {instance_id}: {e}")
        log.error(traceback.format_exc())
        log_data += f"\n--- RESULT: UNEXPECTED ERROR ---\n{traceback.format_exc()}"
        passing_tests = [] # Treat unexpected errors as test failure
    finally:
        # Always try to write the log file
        try:
            with open(log_file_path, "w", encoding='utf-8') as f:
                f.write(log_data)
            log.info(f"Docker execution log saved to: {log_file_path}")
        except IOError as e:
            log.error(f"Failed to write execution log to {log_file_path}: {e}")

        end_time = time.time()
        log.info(f"--- Docker finished for instance: {instance_id} (Duration: {end_time - start_time:.2f}s) ---")

    return passing_tests

# --- Helper Function: find_result_file ---
def find_result_file(output_dir: str, instance_id: str) -> Optional[str]:
    """Finds the .jsonl file containing the specified instance_id."""
    log.info(f"Searching for instance_id '{instance_id}' in directory '{output_dir}'...")
    # Prioritize exact match if instance_id format is predictable
    # e.g., if instance_id is google__gson_abc123 search for google_gson_abc123.jsonl
    try:
         parts = instance_id.split('__')
         if len(parts) == 2:
              org_repo_hash = parts[1] # e.g., gson_abc123
              expected_filename = f"{parts[0]}_{org_repo_hash}.jsonl" # google_gson_abc123.jsonl
              expected_filepath = os.path.join(output_dir, expected_filename)
              if os.path.exists(expected_filepath):
                   log.info(f"Found expected file directly: {expected_filepath}")
                   # Verify content quickly
                   try:
                       with open(expected_filepath, 'r', encoding='utf-8') as f:
                           first_line = f.readline()
                           data = json.loads(first_line)
                           if data.get("instance_id") == instance_id:
                               return expected_filepath
                           else:
                                log.warning(f"Directly found file {expected_filepath} but instance_id inside did not match '{instance_id}'. Continuing search.")
                   except Exception:
                        log.warning(f"Error verifying content of directly found file {expected_filepath}. Continuing search.")

    except Exception: # Ignore errors during predictive search
         pass

    # Fallback to searching all jsonl files
    log.info(f"Falling back to searching content of all .jsonl files in {output_dir}")
    for filename in glob.glob(os.path.join(output_dir, "*.jsonl")):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        if data.get("instance_id") == instance_id:
                            log.info(f"Found instance_id '{instance_id}' in file: {filename} (line {line_num+1})")
                            return filename
                    except json.JSONDecodeError:
                        # Only warn once per file perhaps? Or maybe not at all unless debugging
                        if line_num == 0: log.debug(f"Skipping invalid JSON line in file {filename}: {line.strip()}")
                        continue
                    except Exception:
                         if line_num == 0: log.warning(f"Error reading line data from {filename}", exc_info=True)
                         continue
        except IOError as e:
            log.warning(f"Could not read file {filename}: {e}")
        except Exception as e:
             log.warning(f"An unexpected error occurred processing file {filename}: {e}")

    log.error(f"Could not find any .jsonl file containing instance_id '{instance_id}' in {output_dir}")
    return None

# --- Helper Function: load_original_results ---
def load_original_results(result_file_path: str) -> Optional[Dict]:
    """Loads the first valid JSON object containing required keys from the result file."""
    if not result_file_path or not os.path.exists(result_file_path):
        log.error(f"Result file path is invalid or does not exist: {result_file_path}")
        return None
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
             for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    # Check for essential keys
                    required_keys = ["repo_url", "commit_hash", "tests_passing_in_original_repo", "used_test_command", "used_docker_image", "instance_id"]
                    if all(k in data for k in required_keys):
                        log.info(f"Successfully loaded results from {result_file_path} (line {line_num+1})")
                        return data
                    else:
                        missing_keys = [k for k in required_keys if k not in data]
                        log.warning(f"Skipping line {line_num+1} in {result_file_path} due to missing keys: {missing_keys}. Content: {line.strip()}")
                except json.JSONDecodeError:
                    log.warning(f"Skipping invalid JSON line {line_num+1} in file {result_file_path}: {line.strip()}")
                    continue
                except Exception:
                    log.warning(f"Error reading data from line {line_num+1} in {result_file_path}", exc_info=True)
                    continue
        log.error(f"Could not find valid result data with all required keys in file {result_file_path}")
        return None
    except IOError as e:
        log.error(f"Could not open or read result file {result_file_path}: {e}")
        return None
    except Exception as e:
        log.error(f"An unexpected error occurred loading results from {result_file_path}: {e}")
        return None
# (Includes previous helper functions: clone_and_checkout, run_tests_in_docker,
#  parse_maven_surefire_stdout, parse_gradle_stdout, find_result_file, load_original_results)
# ... (Keep all the previous imports and function definitions) ...

import os
import subprocess
import re
import tempfile
import shutil
import time
import traceback
import json
import logging
import glob
from typing import List, Optional, Tuple, Dict, Set

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(funcName)s] %(message)s'
)
log = logging.getLogger(__name__)

# --- PREVIOUSLY DEFINED HELPER FUNCTIONS ---
# Assume parse_maven_surefire_stdout, parse_gradle_stdout,
# clone_and_checkout, run_tests_in_docker, find_result_file,
# load_original_results are defined above this point, exactly as
# in the previous response where they were included.
# ... (paste those functions here) ...
# --- END OF PREVIOUSLY DEFINED HELPER FUNCTIONS ---


# --- NEW Helper Function: Parse Search/Replace Block ---
def parse_search_replace_block(block_string: str) -> Optional[Tuple[str, str, str]]:
    """
    Parses a single search/replace block string.

    Args:
        block_string: The string containing the ### filepath, SEARCH, and REPLACE sections.

    Returns:
        A tuple (filepath, search_content, replace_content) or None if parsing fails.
    """
    try:
        lines = block_string.strip().split('\n')
        if not lines[0].startswith("### "):
            log.error(f"Block parsing failed: Expected '### filepath' header, got '{lines[0]}'")
            return None
        filepath = lines[0][4:].strip() # Remove "### "

        search_start_index = -1
        separator_index = -1
        replace_end_index = -1

        for i, line in enumerate(lines):
            if line.strip() == "<<<<<<< SEARCH":
                search_start_index = i + 1
            elif line.strip() == "=======":
                separator_index = i
            elif line.strip() == ">>>>>>> REPLACE":
                replace_end_index = i
                break # Found the end

        if not (0 < search_start_index < separator_index < replace_end_index):
             log.error("Block parsing failed: Could not find SEARCH, =======, or REPLACE markers in order.")
             log.debug(f"Indices found: search_start={search_start_index}, separator={separator_index}, replace_end={replace_end_index}")
             return None

        # Extract content, preserving original newlines within blocks
        # The newline *before* ======= belongs to search, *before* >>>>>>> belongs to replace
        search_content = "\n".join(lines[search_start_index:separator_index])
        replace_content = "\n".join(lines[separator_index + 1:replace_end_index])

        return filepath, search_content, replace_content

    except Exception as e:
        log.error(f"Unexpected error parsing search/replace block: {e}", exc_info=True)
        log.error(f"Block content was:\n---\n{block_string}\n---")
        return None


# --- Main Regression Test Function (MODIFIED) ---
def test_patch_regression(
    instance_id: str,
    patch_data: Dict, # MODIFIED: Expects the dictionary with search_replace_blocks
    output_dir: str,
    timeout: int = 1800,
    run_id_prefix: str = "reg_test"
) -> bool:
    """
    Tests if a patch (in search/replace format) causes regressions.

    Args:
        instance_id: The unique identifier for the test case.
        patch_data: A dictionary containing "instance_id" and "search_replace_blocks".
        output_dir: The directory where the original .jsonl result files are stored.
        timeout: Timeout in seconds for the Docker test execution.
        run_id_prefix: Prefix for the run ID used in logging directories.

    Returns:
        True if no originally passing test fails after applying the patch.
        False otherwise.
    """
    log.info(f"--- Starting Regression Test for Instance: {instance_id} (Search/Replace Patch) ---")

    # --- Step 1: Find and Load Original Results ---
    result_file = find_result_file(output_dir, instance_id)
    if not result_file:
        log.error(f"[{instance_id}] Failed to find original result file. Cannot proceed.")
        return False
    original_results = load_original_results(result_file)
    if not original_results:
         log.error(f"[{instance_id}] Failed to load valid data from result file {result_file}. Cannot proceed.")
         return False

    repo_url = original_results.get("repo_url")
    commit_hash = original_results.get("commit_hash")
    tests_passing_original_list = original_results.get("tests_passing_in_original_repo", [])
    test_command = original_results.get("used_test_command")
    docker_image = original_results.get("used_docker_image")
    if not all([repo_url, commit_hash, test_command, docker_image]):
         log.error(f"[{instance_id}] Missing essential information in loaded data.")
         return False
    tests_passing_original = set(tests_passing_original_list)
    log.info(f"[{instance_id}] Original baseline loaded ({len(tests_passing_original)} passing tests).")

    # --- Step 2: Setup Temporary Environment ---
    temp_dir = None
    patch_test_instance_id = f"{instance_id}_patch"
    patch_test_run_id = f"{run_id_prefix}_{int(time.time())}"
    try:
        safe_instance_id_part = re.sub(r'[<>:"/\\|?*]', '_', instance_id)
        temp_dir = tempfile.mkdtemp(prefix=f"regtest_{safe_instance_id_part}_")
        log.info(f"[{instance_id}] Created temporary directory: {temp_dir}")

        # --- Step 3: Clone and Checkout Base Commit ---
        if not clone_and_checkout(repo_url, commit_hash, temp_dir):
            log.error(f"[{instance_id}] Failed to clone repo or checkout base commit. Regression test failed.")
            return False

        # --- Step 4: Apply Patch (MODIFIED LOGIC) ---
        search_replace_blocks = patch_data.get("search_replace_blocks")
        if not isinstance(search_replace_blocks, list):
             log.error(f"[{instance_id}] Invalid patch data: 'search_replace_blocks' is missing or not a list.")
             return False

        log.info(f"[{instance_id}] Applying {len(search_replace_blocks)} search/replace blocks...")
        applied_count = 0
        for i, block_str in enumerate(search_replace_blocks):
            log.info(f"Processing block {i+1}/{len(search_replace_blocks)}...")
            parsed_block = parse_search_replace_block(block_str)
            if not parsed_block:
                log.error(f"[{instance_id}] Failed to parse block {i+1}. Aborting patch application.")
                return False # Treat parse failure as regression failure

            rel_filepath, search_content, replace_content = parsed_block
            abs_filepath = os.path.join(temp_dir, rel_filepath)
            log.info(f"Target file: {abs_filepath}")

            if not os.path.exists(abs_filepath):
                log.error(f"[{instance_id}] Target file specified in block {i+1} does not exist: {abs_filepath}. Aborting.")
                return False

            try:
                # Read original content
                with open(abs_filepath, 'r', encoding='utf-8', errors='replace') as f:
                    original_content = f.read()

                # Perform replacement (only first occurrence)
                if search_content not in original_content:
                    log.error(f"[{instance_id}] SEARCH content from block {i+1} not found in file {rel_filepath}. Patch mismatch. Aborting.")
                    log.debug(f"--- SEARCH CONTENT (Block {i+1}) ---\n{search_content}\n--- END SEARCH ---")
                    # Maybe log snippet of actual file content for comparison?
                    # log.debug(f"--- ACTUAL CONTENT (Snippet) ---\n{original_content[:500]}\n--- END SNIPPET ---")
                    return False

                modified_content = original_content.replace(search_content, replace_content, 1)

                # Write modified content back
                with open(abs_filepath, 'w', encoding='utf-8') as f:
                    f.write(modified_content)

                applied_count += 1
                log.info(f"Successfully applied block {i+1} to {rel_filepath}")

            except IOError as e:
                log.error(f"[{instance_id}] File I/O error processing block {i+1} for file {rel_filepath}: {e}")
                return False
            except Exception as e:
                 log.error(f"[{instance_id}] Unexpected error processing block {i+1} for file {rel_filepath}: {e}", exc_info=True)
                 return False

        log.info(f"[{instance_id}] Finished applying {applied_count} blocks.")
        if applied_count != len(search_replace_blocks):
             log.error(f"[{instance_id}] Mismatch: Expected to apply {len(search_replace_blocks)} blocks, but only applied {applied_count}. This indicates an earlier error.")
             # This case should ideally be caught by earlier returns, but acts as a safeguard
             return False


        # --- Step 5: Run Tests (Post-Patch) ---
        log.info(f"[{instance_id}] Running tests with patch applied...")
        passing_tests_after_patch_list = run_tests_in_docker(
            local_repo_path=temp_dir,
            commit_hash=commit_hash,
            test_command=test_command,
            instance_id=patch_test_instance_id,
            docker_image=docker_image,
            timeout=timeout,
            run_id=patch_test_run_id
        )
        if passing_tests_after_patch_list is None:
            log.error(f"[{instance_id}] Test execution in docker failed to return results. Regression test failed.")
            return False
        tests_passing_after_patch = set(passing_tests_after_patch_list)
        log.info(f"[{instance_id}] {len(tests_passing_after_patch)} tests passed after patch.")

        # --- Step 6: Compare Results ---
        failed_originally_passing = tests_passing_original - tests_passing_after_patch
        if not failed_originally_passing:
            log.info(f"[{instance_id}] SUCCESS: No originally passing tests failed after applying the patch.")
            return True
        else:
            log.error(f"[{instance_id}] FAILURE: {len(failed_originally_passing)} originally passing test(s) failed after applying the patch:")
            max_log_failures = 20
            count = 0
            for test_name in sorted(list(failed_originally_passing)):
                log.error(f"  - {test_name}")
                count += 1
                if count >= max_log_failures:
                    log.error(f"  ... (and {len(failed_originally_passing) - count} more)")
                    break
            return False

    except Exception as e:
        log.error(f"[{instance_id}] An unexpected error occurred during regression test execution: {e}", exc_info=True)
        return False
    finally:
        # --- Step 7: Cleanup ---
        if temp_dir and os.path.exists(temp_dir):
            try:
                log.info(f"[{instance_id}] Cleaning up temporary directory: {temp_dir}")
                def remove_readonly(func, path, _):
                     log.debug(f"Attempting to remove readonly: {path}")
                     os.chmod(path, 0o777)
                     func(path)
                shutil.rmtree(temp_dir, onerror=remove_readonly if os.name == 'nt' else None)
                log.info(f"[{instance_id}] Cleaned up temporary directory.")
            except Exception as e:
                 log.warning(f"[{instance_id}] Unexpected error during cleanup of {temp_dir}: {e}")
        log.info(f"--- Finished Regression Test for Instance: {instance_id} ---")
