# run_reproduction_tests.py

import argparse
import json
import os
import subprocess
import shutil
import logging
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed # Use ProcessPool for isolation
import posixpath # For consistent path splitting regardless of OS

from datasets import load_dataset # To load repo/commit info
# Use the same load_jsonl as in generate_reproduction_tests.py
# (Or import if it's now in a shared util file)
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
    return data

# run_command remains the same as you provided
def run_command(command, working_dir, timeout=None, shell=False):
    """Runs a shell command and captures output, with enhanced logging."""
    log_prefix = f"[{os.path.basename(working_dir)}]" if working_dir else "[UnknownDir]"
    logging.info(f"{log_prefix} Attempting to run command: {' '.join(command)}")
    logging.info(f"{log_prefix} Target working directory: {working_dir}")

    if not working_dir or not os.path.exists(working_dir):
         logging.error(f"{log_prefix} Working directory does NOT exist or is invalid: '{working_dir}'")
         return "", f"Working directory not found: {working_dir}", -4, False # Custom error code
    if not os.path.isdir(working_dir):
         logging.error(f"{log_prefix} Working directory path is not a directory: '{working_dir}'")
         return "", f"Working directory path is not a directory: {working_dir}", -5, False # Custom error code

    if len(command) >= 2 and command[0] == 'git' and command[1] == 'checkout':
        dot_git_path = os.path.join(working_dir, '.git')
        if os.path.exists(dot_git_path) and os.path.isdir(dot_git_path):
             logging.info(f"{log_prefix} Found .git directory at: {dot_git_path}")
        else:
             logging.error(f"{log_prefix} CRITICAL: .git directory NOT FOUND at expected location: {dot_git_path}")
             try:
                 contents = os.listdir(working_dir)
                 logging.error(f"{log_prefix} Contents of '{working_dir}': {contents}")
             except Exception as list_err:
                 logging.error(f"{log_prefix} Could not list contents of '{working_dir}': {list_err}")
             return "", f".git directory not found in {working_dir}", -6, False # Custom error code

    logging.debug(f"Executing command: {' '.join(command)} in {working_dir}")
    try:
        process = subprocess.run(
            command,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            shell=shell
        )
        logging.debug(f"Command finished with return code: {process.returncode}")
        logging.debug(f"stdout:\n{process.stdout[:500]}{'...' if len(process.stdout)>500 else ''}")
        logging.debug(f"stderr:\n{process.stderr[:500]}{'...' if len(process.stderr)>500 else ''}")
        return process.stdout, process.stderr, process.returncode, False # Not timed out
    except subprocess.TimeoutExpired as e:
        logging.warning(f"Command timed out after {timeout}s: {' '.join(command)}")
        stdout = e.stdout.decode(encoding='utf-8', errors='replace') if e.stdout else ""
        stderr = e.stderr.decode(encoding='utf-8', errors='replace') if e.stderr else ""
        return stdout, stderr, -1, True # Indicate timeout with return code -1, True
    except Exception as e:
        logging.error(f"Command failed: {' '.join(command)}\nError: {e}", exc_info=True)
        return "", str(e), -2, False # Indicate other exception

# --- Modified setup_workspace ---
def setup_workspace(instance_id, repair_patch_index, repo_slug, base_commit, base_workspace_dir, run_id):
    """Clones repo and checks out specific commit in a dedicated directory for a specific repair attempt."""
    # Include repair_patch_index for unique directory per attempt
    instance_workspace = os.path.join(base_workspace_dir, f"{instance_id}_idx{repair_patch_index}_{run_id}")
    repo_url = f"https://github.com/{repo_slug}.git" # Assume GitHub
    log_prefix = f"[{instance_id}_idx{repair_patch_index}]"

    logging.info(f"{log_prefix} Setting up workspace in: {instance_workspace}")

    if os.path.exists(instance_workspace):
        logging.warning(f"{log_prefix} Workspace already exists, removing: {instance_workspace}")
        try:
            shutil.rmtree(instance_workspace)
        except OSError as e_clean:
            logging.error(f"{log_prefix} Failed to remove existing workspace {instance_workspace}: {e_clean}")
            return None, f"CLEANUP_FAILED: {e_clean}"

    try:
        os.makedirs(instance_workspace)
        logging.info(f"{log_prefix} Created empty workspace directory: {instance_workspace}")
    except OSError as e_mkdir:
        logging.error(f"{log_prefix} Failed to create workspace directory {instance_workspace}: {e_mkdir}")
        return None, f"MKDIR_FAILED: {e_mkdir}"

    logging.info(f"{log_prefix} Attempting clone of {repo_url} inside {instance_workspace}...")
    cmd_clone = ["git", "clone", repo_url, "."]
    stdout_c, stderr_c, retcode_c, timed_out_c = run_command(cmd_clone, instance_workspace, timeout=300)

    if retcode_c != 0 or timed_out_c:
        logging.error(f"{log_prefix} Failed to clone repository. Ret: {retcode_c}, Timed Out: {timed_out_c}\nStderr: {stderr_c}")
        return None, f"CLONE_FAILED (Ret: {retcode_c}, Timeout: {timed_out_c})"
    if stderr_c and stderr_c.strip():
         logging.warning(f"{log_prefix} Git clone completed with warnings/output on stderr:\n{stderr_c}")

    time.sleep(0.5)
    dot_git_path = os.path.join(instance_workspace, '.git')
    if not os.path.exists(instance_workspace):
         logging.error(f"{log_prefix} CRITICAL: Target workspace directory '{instance_workspace}' missing after clone attempt!")
         return None, "CLONE_VERIFY_FAILED (Workspace dir missing)"
    if not os.path.exists(dot_git_path) or not os.path.isdir(dot_git_path):
         logging.error(f"{log_prefix} CRITICAL: .git directory NOT FOUND at '{dot_git_path}' after supposedly successful clone!")
         try:
             contents = os.listdir(instance_workspace)
             logging.error(f"{log_prefix} Contents of '{instance_workspace}' after clone attempt: {contents}")
         except Exception as list_err:
             logging.error(f"{log_prefix} Could not list contents of '{instance_workspace}' after clone attempt: {list_err}")
         return None, "CLONE_VERIFY_FAILED (.git dir missing)"

    logging.info(f"{log_prefix} Clone appears successful. Proceeding to checkout.")

    logging.info(f"{log_prefix} Checking out commit {base_commit}...")
    cmd_checkout = ["git", "checkout", base_commit]
    stdout_co, stderr_co, retcode_co, timed_out_co = run_command(cmd_checkout, instance_workspace, timeout=60)
    if retcode_co != 0 or timed_out_co:
        logging.error(f"{log_prefix} Failed to checkout commit {base_commit}. Ret: {retcode_co}, Timed Out: {timed_out_co}\nStderr: {stderr_co}")
        return None, f"CHECKOUT_FAILED (Ret: {retcode_co}, Timeout: {timed_out_co})"

    logging.info(f"{log_prefix} Workspace setup complete.")
    return instance_workspace, None # Return path and no error message

# --- REMOVED parse_search_replace_block ---
# --- REMOVED apply_search_replace ---

# --- NEW Helper Function ---
def extract_module_from_diff(diff_content, instance_id, repair_patch_index):
    """
    Parses the first 'diff --git a/path...' line to infer the target module.
    Uses posixpath for splitting to handle paths correctly.
    """
    log_prefix = f"[{instance_id}_idx{repair_patch_index}]"
    try:
        first_line = diff_content.split('\n', 1)[0]
        if first_line.startswith('diff --git a/'):
            parts = first_line.split(' ')
            if len(parts) >= 3 and parts[2].startswith('a/'):
                # Use posixpath to handle separators correctly
                file_path = parts[2][2:] # Get path after 'a/'
                path_parts = file_path.split(posixpath.sep)
                if len(path_parts) > 1:
                    target_module = path_parts[0]
                    logging.info(f"{log_prefix} Inferred target module '{target_module}' from path '{file_path}'")
                    return target_module
                else:
                    logging.info(f"{log_prefix} Path '{file_path}' has no directory components. Assuming no specific module.")
                    return None # No module subdir detected
            else:
                 logging.warning(f"{log_prefix} Could not parse 'diff --git' line format: {first_line}")
                 return None
        else:
            logging.warning(f"{log_prefix} Repair patch does not start with 'diff --git a/': {first_line[:100]}...")
            return None
    except Exception as e:
        logging.error(f"{log_prefix} Error extracting module from diff: {e}")
        return None


# --- Modified apply_diff_patch ---
def apply_diff_patch(patch_content, repo_dir, instance_id, repair_patch_index, patch_type="Unknown"):
    """
    Applies a standard git diff patch.

    Args:
        patch_content (str): The content of the Git diff patch.
        repo_dir (str): The path to the repository workspace.
        instance_id (str): For logging.
        repair_patch_index (int): For logging.
        patch_type (str): Description ("Repair" or "Test").

    Returns:
        str: None if successful, otherwise an error message.
    """
    log_prefix = f"[{instance_id}_idx{repair_patch_index}]"
    if not patch_content or not patch_content.strip():
        logging.warning(f"{log_prefix} Provided {patch_type} patch content is empty. Skipping application.")
        # Return specific code if it's the test patch that's missing
        if patch_type == "Test":
            return "TEST_PATCH_EMPTY"
        return None # Empty repair patch might be okay? Or should be an error? Assume ok for now.

    try:
        # Use utf-8 explicitly, handle potential BOM
        if patch_content.startswith('\ufeff'):
             logging.warning(f"{log_prefix} Removing BOM from {patch_type} patch")
             patch_content = patch_content[1:]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".diff", encoding='utf-8') as temp_patch_file:
            temp_patch_file.write(patch_content)
            temp_patch_path = temp_patch_file.name
        logging.info(f"{log_prefix} Applying {patch_type} patch from temporary file: {temp_patch_path}")

        # Use git apply
        # Consider adding --verbose for more debug info if needed
        cmd_apply = ["git", "apply", "--whitespace=fix", temp_patch_path]
        stdout, stderr, retcode, timed_out = run_command(cmd_apply, repo_dir, timeout=60)

        if retcode != 0:
            error_msg = f"PATCH_APPLY_FAILED ({patch_type} Patch): `git apply` failed with code {retcode}.\nStderr: {stderr}\nStdout: {stdout}"
            logging.error(f"{log_prefix} {error_msg}")
            # Try to provide context from the patch
            logging.error(f"{log_prefix} Failing {patch_type} Patch Content (first 500 chars):\n{patch_content[:500]}")
            return error_msg
        else:
            logging.info(f"{log_prefix} Successfully applied {patch_type} patch.")
            # Log stderr warnings if any
            if stderr and stderr.strip():
                logging.warning(f"{log_prefix} `git apply` for {patch_type} patch completed with warnings/output on stderr:\n{stderr}")
            return None # Success

    except Exception as e:
        error_msg = f"PATCH_APPLY_EXCEPTION ({patch_type} Patch): Unexpected error applying patch: {e}"
        logging.error(f"{log_prefix} {error_msg}", exc_info=True)
        return error_msg
    finally:
        if 'temp_patch_path' in locals() and os.path.exists(temp_patch_path):
            try:
                os.remove(temp_patch_path)
            except OSError as e_clean:
                logging.warning(f"{log_prefix} Failed to remove temporary patch file {temp_patch_path}: {e_clean}")


# run_maven_test remains the same as you provided
def run_maven_test(repo_dir, test_class_name, timeout, instance_id, repair_patch_index, target_module=None):
    """
    Runs a specific test class using Maven. Includes -U flag and optionally targets a module.
    """
    log_prefix = f"[{instance_id}_idx{repair_patch_index}]"
    if not os.path.exists(os.path.join(repo_dir, "pom.xml")):
        logging.error(f"{log_prefix} No pom.xml found in {repo_dir}. Cannot run Maven.")
        return "", "No pom.xml found", -3, False # Custom error code

    logging.info(f"{log_prefix} Attempting Maven compilation (including test sources)...")
    cmd_compile = [ "mvn", "-B", "-ntp", "-U", "clean", "test-compile", "-e" ]
    if target_module:
        # Ensure module format is correct (e.g., group:artifact or just artifact)
        # Simple approach: assume artifact ID is sufficient if it's just one segment
        if ':' not in target_module and '/' not in target_module:
             module_arg = f":{target_module}" # Common for single module project structure
        else:
             module_arg = target_module # Assume user provided correct format
        cmd_compile.extend(["-pl", module_arg, "-am"])
        logging.info(f"{log_prefix} Targeting module {module_arg} for compilation.")

    compile_timeout = timeout / 2 if timeout else None
    stdout_compile, stderr_compile, retcode_compile, timed_out_compile = run_command(
        cmd_compile, repo_dir, timeout=compile_timeout
    )

    if timed_out_compile:
        logging.error(f"{log_prefix} Maven compilation timed out after {compile_timeout}s.")
        return stdout_compile, stderr_compile, -1, True # Report timeout
    if retcode_compile != 0:
        if "Non-resolvable parent POM" in stdout_compile or "Could not resolve dependencies" in stdout_compile or "Could not find artifact" in stderr_compile:
            logging.error(f"{log_prefix} Maven build failed due to dependency resolution error (likely missing SNAPSHOT or incorrect repo setup). RC: {retcode_compile}")
            logging.error(f"{log_prefix} Relevant compilation stdout snippet:\n{stdout_compile[-1500:]}") # Log end of stdout
            logging.error(f"{log_prefix} Relevant compilation stderr snippet:\n{stderr_compile[-1500:]}") # Log end of stderr
        else:
            logging.error(f"{log_prefix} Maven compilation failed with return code {retcode_compile}.")
            logging.error(f"{log_prefix} Compilation stdout:\n{stdout_compile}")
            logging.error(f"{log_prefix} Compilation stderr:\n{stderr_compile}")
        return stdout_compile, stderr_compile, retcode_compile, False # Return specific failure

    logging.info(f"{log_prefix} Maven compilation successful. Proceeding to test execution.")

    test_pattern = f"**/{test_class_name}.java" # Be more specific for Surefire
    logging.info(f"{log_prefix} Using test pattern for Surefire: {test_pattern}")

    cmd_test = [
        "mvn", "-B", "-ntp", "-U",
        "org.apache.maven.plugins:maven-surefire-plugin:3.2.5:test", # Use specific goal
        f"-Dtest={test_pattern}",
        "-DfailIfNoTests=false",
        "-Dmaven.test.failure.ignore=true", # <<< --- ADDED: Don't fail build on test failure ---
        "-Dsurefire.failIfNoSpecifiedTests=false", # <<< --- ADDED: Don't fail if test class not found ---
        "-DjacocoArgLine= "
    ]

    if target_module:
         if ':' not in target_module and '/' not in target_module:
             module_arg = f":{target_module}"
         else:
             module_arg = target_module
         cmd_test.extend(["-pl", module_arg]) # Target the specific module for test execution
         logging.info(f"{log_prefix} Targeting module {module_arg} for test execution.")

    logging.info(f"{log_prefix} Running Maven Surefire test command: {' '.join(cmd_test)}")
    test_timeout = timeout - compile_timeout if timeout and compile_timeout else timeout
    stdout_test, stderr_test, retcode_test, timed_out_test = run_command(
        cmd_test, repo_dir, timeout=test_timeout
    )

    # Note: We ignore retcode_test here because we added -Dmaven.test.failure.ignore=true
    # We will rely solely on the stdout/stderr markers for the outcome.
    if timed_out_test:
        logging.warning(f"{log_prefix} Maven test execution timed out after {test_timeout}s.")
    elif retcode_test != 0:
         logging.warning(f"{log_prefix} Maven Surefire execution finished with non-zero return code {retcode_test} (but failure ignored).")
         logging.warning(f"{log_prefix} Surefire stderr (if any):\n{stderr_test}")
    else:
        logging.info(f"{log_prefix} Maven Surefire execution completed (Return Code 0).")

    combined_stdout = f"--- Compile Output ---\n{stdout_compile}\n--- Test Output ---\n{stdout_test}"
    combined_stderr = f"--- Compile Stderr ---\n{stderr_compile}\n--- Test Stderr ---\n{stderr_test}"

    # Return the combined results, but indicate success based on whether timeout occurred
    # The actual test pass/fail/reproduce status is determined by analyze_test_output
    return combined_stdout, combined_stderr, 0 if not timed_out_test else -1, timed_out_test


# analyze_test_output remains the same as you provided
def analyze_test_output(stdout, stderr, return_code, timeout_expired, instance_id, repair_patch_index):
    """
    Determines the outcome based on test execution results (stdout/stderr/return code).
    Focuses on markers printed by the test.
    """
    log_prefix = f"[{instance_id}_idx{repair_patch_index}]"
    logging.info(f"{log_prefix} Analyzing test output. Return code: {return_code}, Timed out: {timeout_expired}")

    # Combine stdout and stderr for searching marker strings
    full_output = stdout + "\n" + stderr

    # Check for timeout first
    if timeout_expired:
        logging.warning(f"{log_prefix} Outcome: TEST_TIMEOUT")
        return "TEST_TIMEOUT"

    # Check for explicit markers in the output (prioritize RESOLVED)
    # Use find() for efficiency and check result isn't -1
    if full_output.find("ISSUE_RESOLVED") != -1:
        logging.info(f"{log_prefix} Outcome: RESOLVED (found 'ISSUE_RESOLVED')")
        return "RESOLVED"
    elif full_output.find("ISSUE_REPRODUCED") != -1:
        logging.info(f"{log_prefix} Outcome: REPRODUCED (found 'ISSUE_REPRODUCED')")
        return "REPRODUCED"
    elif full_output.find("OTHER_ISSUES") != -1:
         logging.warning(f"{log_prefix} Outcome: OTHER_ISSUES (found 'OTHER_ISSUES')")
         return "OTHER_ISSUES"

    # --- Error Analysis (No Markers Found) ---
    # Did the test run at all? Check Surefire output.
    if "Tests run: 0" in full_output or "No tests were executed!" in full_output:
        outcome = "TEST_ERROR (No tests found/run)"
        logging.error(f"{log_prefix} Outcome: {outcome}. Surefire did not find/run the test class.")
        logging.error(f"{log_prefix} Maven stdout for diagnosis:\n------ STDOUT START ------\n{stdout}\n------ STDOUT END ------")
        logging.error(f"{log_prefix} Maven stderr for diagnosis:\n------ STDERR START ------\n{stderr}\n------ STDERR END ------")
        return outcome

    # Check for compilation errors within the test execution phase (less likely if compile step passed)
    if "COMPILATION ERROR" in full_output:
        outcome = "TEST_ERROR (Compilation failed)"
        logging.error(f"{log_prefix} Outcome: {outcome}. Found compilation error during test phase.")
        logging.error(f"{log_prefix} Maven stdout for diagnosis:\n------ STDOUT START ------\n{stdout}\n------ STDOUT END ------")
        logging.error(f"{log_prefix} Maven stderr for diagnosis:\n------ STDERR START ------\n{stderr}\n------ STDERR END ------")
        return outcome

    # Generic error if no markers and no specific failure pattern identified
    outcome = "TEST_ERROR (No markers, RC=0)" # RC=0 because we ignore surefire failures now
    logging.warning(f"{log_prefix} Outcome: {outcome} (Command finished but no expected markers found)")
    logging.warning(f"{log_prefix} Maven stdout for ambiguous diagnosis:\n------ STDOUT START ------\n{stdout}\n------ STDOUT END ------")
    logging.warning(f"{log_prefix} Maven stderr for ambiguous diagnosis:\n------ STDERR START ------\n{stderr}\n------ STDERR END ------")
    return outcome


# --- Modified run_single_instance_test ---
def run_single_repair_attempt(attempt_data, args):
    """
    Orchestrates the testing process for a single repair patch attempt.
    """
    instance_id = attempt_data['instance_id']
    repair_patch_index = attempt_data['repair_patch_index'] # Unique index for this attempt
    repo_slug = attempt_data['repo']
    base_commit = attempt_data['base_commit']
    repair_patch_content = attempt_data['repair_patch_content'] # Git diff format
    test_patch_content = attempt_data['test_patch_content'] # Git diff format to add test

    log_prefix = f"[{instance_id}_idx{repair_patch_index}]"
    instance_workspace = None # Initialize for cleanup
    final_outcome = "UNKNOWN_ERROR" # Default

    try:
        # 1. Setup Workspace (using unique index)
        logging.info(f"{log_prefix} Starting test run for repair index {repair_patch_index}.")
        instance_workspace, setup_error = setup_workspace(
            instance_id, repair_patch_index, repo_slug, base_commit, args.workspace_dir, args.run_id
        )
        if setup_error:
            logging.error(f"{log_prefix} Workspace setup failed: {setup_error}")
            return f"SETUP_FAILED: {setup_error}" # Return error outcome string

        # --- NEW: Extract target module BEFORE applying patch ---
        target_module = extract_module_from_diff(repair_patch_content, instance_id, repair_patch_index)
        if not target_module:
             logging.warning(f"{log_prefix} Could not determine target module from repair patch. Will run test from root.")
        # --- END NEW ---

        # 2. Apply Repair Patch (Git Diff)
        logging.info(f"{log_prefix} Applying repair patch...")
        apply_repair_error = apply_diff_patch(repair_patch_content, instance_workspace, instance_id, repair_patch_index, patch_type="Repair")
        if apply_repair_error:
            logging.error(f"{log_prefix} Failed to apply repair patch: {apply_repair_error}")
            return f"REPAIR_PATCH_FAILED: {apply_repair_error}"
        logging.info(f"{log_prefix} Repair patch applied successfully.")

        # Check for Empty Test Patch BEFORE Applying/Running
        if not test_patch_content or not test_patch_content.strip():
            logging.warning(f"{log_prefix} Test patch content is empty (from file {args.generated_test_file}). Test generation likely failed for {instance_id}.")
            final_outcome = "TEST_GENERATION_FAILED"
            return final_outcome

        # 3. Apply Test Patch (Git Diff)
        logging.info(f"{log_prefix} Applying generated test patch...")
        apply_test_patch_error = apply_diff_patch(test_patch_content, instance_workspace, instance_id, repair_patch_index, patch_type="Test")
        # Handle case where test patch itself was empty
        if apply_test_patch_error == "TEST_PATCH_EMPTY":
             logging.warning(f"{log_prefix} Test patch was empty, cannot run test.")
             return "TEST_GENERATION_FAILED" # Treat same as if gen failed
        elif apply_test_patch_error:
            logging.error(f"{log_prefix} Failed to apply test patch: {apply_test_patch_error}")
            return f"TEST_PATCH_FAILED: {apply_test_patch_error}"
        logging.info(f"{log_prefix} Test patch applied successfully.")

        # 4. Run the Test (Passing target module if found)
        logging.info(f"{log_prefix} Running Maven test...")
        test_class_name = "ReproduceBugTest" # Name used in the generation prompt
        stdout, stderr, retcode, timed_out = run_maven_test(
            instance_workspace,
            test_class_name,
            args.timeout,
            instance_id,
            repair_patch_index, # Pass index for logging
            target_module=target_module # Pass the inferred module
        )

        # 5. Analyze Test Output
        logging.info(f"{log_prefix} Analyzing test results...")
        final_outcome = analyze_test_output(stdout, stderr, retcode, timed_out, instance_id, repair_patch_index)
        logging.info(f"{log_prefix} Final outcome for repair index {repair_patch_index}: {final_outcome}")

        return final_outcome # Return the analyzed result string

    except Exception as e:
        logging.error(f"{log_prefix} Unexpected exception during testing: {e}", exc_info=True)
        final_outcome = f"RUNNER_UNEXPECTED_EXCEPTION: {e}"
        return final_outcome
    finally:
        # 6. Cleanup Workspace (Optional)
        if args.cleanup_workspace and instance_workspace and os.path.exists(instance_workspace):
            logging.info(f"{log_prefix} Cleaning up workspace: {instance_workspace}")
            try:
                shutil.rmtree(instance_workspace)
            except OSError as e_clean:
                logging.error(f"{log_prefix} Failed to cleanup workspace {instance_workspace}: {e_clean}")
        elif instance_workspace:
             logging.info(f"{log_prefix} Workspace kept at: {instance_workspace}")


def main():
    parser = argparse.ArgumentParser(description="Run generated Java reproduction tests against code repair patches.")

    # --- Modified Input Files ---
    parser.add_argument("--repair_patch_file", type=str, required=True, help="Path to JSONL file containing repair patches (e.g., filtered_patches.jsonl). Expected fields: instance_id, model_patch, [generation_index/model_name_or_path].")
    parser.add_argument("--generated_test_file", type=str, required=True, help="Path to JSONL file with generated test patches (output_N... file from generate_reproduction_tests.py). Expected fields: instance_id, test_patch.")
    # --- End Modified Input Files ---

    parser.add_argument("--dataset_name", type=str, default="Daoguang/Multi-SWE-bench", help="Name of the Hugging Face dataset (for repo/commit info).")
    parser.add_argument("--dataset_split", type=str, default="java_verified", help="Split of the dataset to use.")
    parser.add_argument("--dataset_cache_dir", type=str, default=None, help="Directory to cache Hugging Face datasets.")

    # Execution Configuration
    parser.add_argument("--workspace_dir", type=str, required=True, help="Base directory to create temporary workspaces for checkouts.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the final validation results (JSONL).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel processes to run tests.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for running the test command (compile + test run).")
    parser.add_argument("--run_id", type=str, default=time.strftime("%Y%m%d_%H%M%S"), help="Identifier for this run (used in logging/temp dirs).")
    parser.add_argument("--instance_ids", nargs='+', default=None, help="Optional list of specific instance_ids to run (will run all repair patches for these IDs).")
    parser.add_argument("--skip_existing", action='store_true', help="Skip attempts (instance_id + repair_index) already present in the results file.")
    parser.add_argument("--cleanup_workspace", action='store_true', help="Remove instance workspace after testing.")

    args = parser.parse_args()

    # Setup logging
    log_file_path = os.path.join(os.path.dirname(args.results_file), f"run_repro_tests_{args.run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Also print to console
        ]
    )
    logging.info(f"Starting run {args.run_id}. Logging to {log_file_path}")
    logging.info(f"Arguments: {vars(args)}")


    # --- Load Data ---
    logging.info("Loading data...")
    all_repair_patches = []
    try:
        loaded_repairs = load_jsonl(args.repair_patch_file)
        # Store with original index and check required fields
        for idx, item in enumerate(loaded_repairs):
             if 'instance_id' in item and 'model_patch' in item:
                 # Add index for unique identification of each repair attempt
                 item['repair_patch_index'] = item.get('generation_index', idx)
                 all_repair_patches.append(item)
             else:
                 logging.warning(f"Skipping entry {idx} in repair patch file due to missing 'instance_id' or 'model_patch': {item}")
        logging.info(f"Loaded {len(all_repair_patches)} repair patch attempts from {args.repair_patch_file}")
    except Exception as e:
        logging.error(f"Failed to load repair patches from {args.repair_patch_file}: {e}", exc_info=True)
        return

    # Load test patches into a dictionary for quick lookup
    test_patches_dict = {}
    try:
        loaded_tests = load_jsonl(args.generated_test_file)
        for item in loaded_tests:
             if 'instance_id' in item and 'test_patch' in item:
                 # Store only the first test patch found for an instance_id
                 if item['instance_id'] not in test_patches_dict:
                     test_patches_dict[item['instance_id']] = item['test_patch']
                 else:
                      logging.warning(f"Multiple test patches found for {item['instance_id']} in {args.generated_test_file}. Using the first one encountered.")
             else:
                 logging.warning(f"Skipping entry in generated test file due to missing 'instance_id' or 'test_patch': {item}")
        logging.info(f"Loaded test patches for {len(test_patches_dict)} unique instances from {args.generated_test_file}")
    except Exception as e:
        logging.error(f"Failed to load test patches from {args.generated_test_file}: {e}", exc_info=True)
        return

    # Load dataset info
    dataset_info_dict = {}
    try:
        logging.info(f"Loading dataset '{args.dataset_name}' split '{args.dataset_split}'...")
        # Use streaming=True for potentially large datasets? For now, load fully.
        dataset = load_dataset(args.dataset_name, split=args.dataset_split, cache_dir=args.dataset_cache_dir, trust_remote_code=True)
        for item in dataset:
             if 'instance_id' in item and 'repo' in item and 'base_commit' in item:
                 dataset_info_dict[item['instance_id']] = {'repo': item['repo'], 'base_commit': item['base_commit']}
             else:
                 logging.warning(f"Skipping dataset entry due to missing fields: {item.get('instance_id', 'MISSING_ID')}")
        logging.info(f"Loaded info for {len(dataset_info_dict)} instances from dataset.")
    except Exception as e:
        logging.error(f"Failed to load dataset info: {e}", exc_info=True)
        return

    # --- Prepare Instance List ---
    # Filter repair patches based on args.instance_ids if provided
    if args.instance_ids:
        target_ids_set = set(args.instance_ids)
        all_repair_patches = [p for p in all_repair_patches if p['instance_id'] in target_ids_set]
        logging.info(f"Filtered repair patches to {len(all_repair_patches)} attempts based on --instance_ids argument.")


    # Create the final list of tasks, merging required data
    tasks_to_run = []
    missing_test_count = 0
    missing_dataset_info_count = 0
    for repair_attempt in all_repair_patches:
        instance_id = repair_attempt['instance_id']
        repair_patch_index = repair_attempt['repair_patch_index']

        test_patch = test_patches_dict.get(instance_id)
        dataset_info = dataset_info_dict.get(instance_id)

        if not test_patch:
            logging.warning(f"[{instance_id}_idx{repair_patch_index}] Skipping: No corresponding test patch found in {args.generated_test_file}.")
            missing_test_count += 1
            continue
        if not dataset_info:
            logging.warning(f"[{instance_id}_idx{repair_patch_index}] Skipping: No corresponding dataset info (repo/commit) found.")
            missing_dataset_info_count += 1
            continue

        tasks_to_run.append({
            'instance_id': instance_id,
            'repair_patch_index': repair_patch_index,
            'repair_patch_content': repair_attempt['model_patch'],
            'test_patch_content': test_patch, # Content directly
            'repo': dataset_info['repo'],
            'base_commit': dataset_info['base_commit'],
            # Include model info if available for results file
            'model_name_or_path': repair_attempt.get('model_name_or_path', 'N/A')
        })

    logging.info(f"Prepared {len(tasks_to_run)} testing tasks.")
    if missing_test_count > 0: logging.warning(f"{missing_test_count} tasks skipped due to missing test patches.")
    if missing_dataset_info_count > 0: logging.warning(f"{missing_dataset_info_count} tasks skipped due to missing dataset info.")

    if not tasks_to_run:
        logging.warning("No valid tasks to run after filtering and data merging. Exiting.")
        return

    # --- Skip Existing ---
    existing_results = {} # Key: tuple(instance_id, repair_patch_index), Value: result item
    if args.skip_existing and os.path.exists(args.results_file):
        logging.info(f"Loading existing results from {args.results_file} to skip...")
        try:
            loaded_existing = load_jsonl(args.results_file)
            for item in loaded_existing:
                # Use a tuple key to handle composite key of instance and index
                if 'instance_id' in item and 'repair_patch_index' in item:
                     existing_results[(item['instance_id'], item['repair_patch_index'])] = item
                else:
                     logging.warning(f"Skipping existing result item due to missing keys: {item}")
            original_count = len(tasks_to_run)
            tasks_to_run = [
                task for task in tasks_to_run
                if (task['instance_id'], task['repair_patch_index']) not in existing_results
            ]
            logging.info(f"Skipped {original_count - len(tasks_to_run)} attempts found in existing results.")
        except Exception as e:
            logging.error(f"Error processing existing results file {args.results_file}: {e}. Proceeding without skipping.", exc_info=True)
            existing_results = {} # Reset on error

    if not tasks_to_run:
        logging.info("No new attempts to run after skipping. Exiting.")
        # Still save the existing results back to the file to ensure it's valid JSONL
        if existing_results:
             try:
                with open(args.results_file, 'w', encoding='utf-8') as f:
                    for entry in existing_results.values():
                         f.write(json.dumps(entry) + '\n')
             except Exception as e:
                 logging.error(f"Failed to rewrite existing results: {e}", exc_info=True)
        return


    # --- Run Tests ---
    os.makedirs(args.workspace_dir, exist_ok=True)
    # Store new results keyed by (instance_id, repair_patch_index)
    new_results = {} # Key: tuple(instance_id, repair_patch_index), Value: outcome string

    logging.info(f"Starting test execution for {len(tasks_to_run)} repair attempts using {args.num_workers} worker(s)...")
    # Use tqdm for progress bar
    from tqdm import tqdm

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit tasks
            future_to_key = {
                executor.submit(run_single_repair_attempt, task_data, args):
                (task_data['instance_id'], task_data['repair_patch_index'])
                for task_data in tasks_to_run
            }
            # Process results as they complete
            for future in tqdm(as_completed(future_to_key), total=len(tasks_to_run), desc="Running Tests"):
                key = future_to_key[future]
                try:
                    outcome = future.result()
                    new_results[key] = outcome
                except Exception as exc:
                    logging.error(f'{key[0]}_idx{key[1]} generated an exception in executor: {exc}', exc_info=True)
                    new_results[key] = "RUNNER_EXCEPTION_MAIN"
    else: # Run sequentially for easier debugging
        logging.info("Running in single-process mode.")
        for task_data in tqdm(tasks_to_run, desc="Running Tests"):
             key = (task_data['instance_id'], task_data['repair_patch_index'])
             try:
                 outcome = run_single_repair_attempt(task_data, args)
                 new_results[key] = outcome
             except Exception as exc:
                 logging.error(f'{key[0]}_idx{key[1]} generated an exception: {exc}', exc_info=True)
                 new_results[key] = "RUNNER_EXCEPTION_MAIN"

    # --- Save Results ---
    logging.info(f"Finished testing. Saving {len(new_results)} new results to {args.results_file}")

    # Combine existing and new results
    final_results_map = {**existing_results, **new_results} # new_results will overwrite existing if re-run

    # Add metadata back for saving
    final_results_list = []
    # Need to find the original task data corresponding to the result key to get metadata back
    all_task_data_map = {(t['instance_id'], t['repair_patch_index']): t for t in tasks_to_run}

    for key, outcome in final_results_map.items():
         # Retrieve original task data if it was run in this session, otherwise use existing
         original_task = all_task_data_map.get(key)
         if original_task:
             final_results_list.append({
                 "instance_id": key[0],
                 "repair_patch_index": key[1],
                 "model_name_or_path": original_task.get('model_name_or_path', 'N/A'),
                 "outcome": outcome,
                 "run_id": args.run_id
             })
         elif key in existing_results:
              # If the result came from the existing file, write it back as is
              final_results_list.append(existing_results[key])
         else:
              # Should not happen if logic is correct, but handle defensively
              logging.warning(f"Could not find original data for result key {key}. Saving minimal info.")
              final_results_list.append({
                   "instance_id": key[0],
                   "repair_patch_index": key[1],
                   "outcome": outcome,
                   "run_id": args.run_id
              })


    try:
        # Sort results for consistency (optional)
        final_results_list.sort(key=lambda x: (x['instance_id'], x['repair_patch_index']))

        with open(args.results_file, 'w', encoding='utf-8') as f:
            for entry in final_results_list:
                f.write(json.dumps(entry) + '\n')
        logging.info("Results saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save results: {e}", exc_info=True)

    logging.info("Script finished.")


if __name__ == "__main__":
    # Add import for tqdm if not already present globally
    try:
         from tqdm import tqdm
    except ImportError:
         print("Warning: tqdm not installed. Progress bars will not be shown.")
         # Define a dummy tqdm if not installed
         def tqdm(iterable, *args, **kwargs):
             return iterable
    main()