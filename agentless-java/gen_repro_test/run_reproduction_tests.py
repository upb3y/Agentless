# run_java_tests.py (or adapted run_reproduction_tests.py)

import argparse
import json
import os
import subprocess
import shutil
import logging
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed # Use ProcessPool for isolation

from datasets import load_dataset # To load repo/commit info
# Use the same load_jsonl as in generate_reproduction_tests.py
# (Or import if it's now in a shared util file)
def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {line.strip()} - Error: {e}")
    return data


def run_command(command, working_dir, timeout=None, shell=False):
    """Runs a shell command and captures output, with enhanced logging."""
    # --- ADDED/ENHANCED LOGGING ---
    # Try to get a meaningful prefix from the directory name
    log_prefix = f"[{os.path.basename(working_dir)}]" if working_dir else "[UnknownDir]"
    logging.info(f"{log_prefix} Attempting to run command: {' '.join(command)}")
    logging.info(f"{log_prefix} Target working directory: {working_dir}")

    # Check if the working directory exists and is valid *before* running
    if not working_dir or not os.path.exists(working_dir):
         logging.error(f"{log_prefix} Working directory does NOT exist or is invalid: '{working_dir}'")
         return "", f"Working directory not found: {working_dir}", -4, False # Custom error code
    if not os.path.isdir(working_dir):
         logging.error(f"{log_prefix} Working directory path is not a directory: '{working_dir}'")
         return "", f"Working directory path is not a directory: {working_dir}", -5, False # Custom error code

    # Specifically check for the .git directory *before* attempting checkout
    # Assumes the command list structure is ['git', 'checkout', ...]
    if len(command) >= 2 and command[0] == 'git' and command[1] == 'checkout':
        dot_git_path = os.path.join(working_dir, '.git')
        if os.path.exists(dot_git_path) and os.path.isdir(dot_git_path):
             logging.info(f"{log_prefix} Found .git directory at: {dot_git_path}")
        else:
             # This is the critical check - if .git isn't here, checkout will fail
             logging.error(f"{log_prefix} CRITICAL: .git directory NOT FOUND at expected location: {dot_git_path}")
             # Log directory contents for debugging clues
             try:
                 contents = os.listdir(working_dir)
                 logging.error(f"{log_prefix} Contents of '{working_dir}': {contents}")
             except Exception as list_err:
                 logging.error(f"{log_prefix} Could not list contents of '{working_dir}': {list_err}")
             # Return an error immediately instead of running the failing git command
             return "", f".git directory not found in {working_dir}", -6, False # Custom error code
    # --- END ADDED/ENHANCED LOGGING ---

    # Original debug logging - keep this too
    logging.debug(f"Executing command: {' '.join(command)} in {working_dir}")
    try:
        process = subprocess.run(
            command,
            cwd=working_dir, # Pass the confirmed working directory
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, # Get stdout/stderr as strings
            encoding='utf-8', # Explicitly set encoding
            errors='replace', # Handle potential decoding errors
            timeout=timeout,
            shell=shell # Be cautious with shell=True
        )
        logging.debug(f"Command finished with return code: {process.returncode}")
        # Log truncated output for brevity
        logging.debug(f"stdout:\n{process.stdout[:500]}{'...' if len(process.stdout)>500 else ''}")
        logging.debug(f"stderr:\n{process.stderr[:500]}{'...' if len(process.stderr)>500 else ''}")
        return process.stdout, process.stderr, process.returncode, False # Not timed out
    except subprocess.TimeoutExpired as e:
        logging.warning(f"Command timed out after {timeout}s: {' '.join(command)}")
        # Access stdout/stderr from the exception object
        stdout = e.stdout.decode(encoding='utf-8', errors='replace') if e.stdout else ""
        stderr = e.stderr.decode(encoding='utf-8', errors='replace') if e.stderr else ""
        return stdout, stderr, -1, True # Indicate timeout with return code -1, True
    except Exception as e:
        # Log the exception with traceback for detailed debugging
        logging.error(f"Command failed: {' '.join(command)}\nError: {e}", exc_info=True)
        return "", str(e), -2, False # Indicate other exception

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')

# --- Function Placeholders (We will implement these) ---

# Replace the existing setup_workspace function in your script with this one
# Make sure imports for os, shutil, logging, time are present

def setup_workspace(instance_id, repo_slug, base_commit, base_workspace_dir, run_id):
    """Clones repo and checks out specific commit in a dedicated directory."""
    instance_workspace = os.path.join(base_workspace_dir, f"{instance_id}_{run_id}")
    repo_url = f"https://github.com/{repo_slug}.git" # Assume GitHub

    logging.info(f"[{instance_id}] Setting up workspace in: {instance_workspace}")

    # Clean previous attempt if exists
    if os.path.exists(instance_workspace):
        logging.warning(f"[{instance_id}] Workspace already exists, removing: {instance_workspace}")
        try:
            shutil.rmtree(instance_workspace)
        except OSError as e_clean:
            logging.error(f"[{instance_id}] Failed to remove existing workspace {instance_workspace}: {e_clean}")
            return None, f"CLEANUP_FAILED: {e_clean}"

    try:
        # Create the directory first
        os.makedirs(instance_workspace)
        logging.info(f"[{instance_id}] Created empty workspace directory: {instance_workspace}")
    except OSError as e_mkdir:
        logging.error(f"[{instance_id}] Failed to create workspace directory {instance_workspace}: {e_mkdir}")
        return None, f"MKDIR_FAILED: {e_mkdir}"


    # --- Clone Strategy Change ---
    # Run 'git clone <url> .' INSIDE the target directory
    logging.info(f"[{instance_id}] Attempting clone of {repo_url} inside {instance_workspace}...")
    # Command clones into the current directory ('.')
    cmd_clone = ["git", "clone", repo_url, "."]
    # Run the clone command *inside* the newly created instance_workspace
    stdout_c, stderr_c, retcode_c, timed_out_c = run_command(cmd_clone, instance_workspace, timeout=300) # Run INSIDE instance_workspace
    # --- End Clone Strategy Change ---


    # --- Enhanced Check after Clone (Same checks as before) ---
    if retcode_c != 0 or timed_out_c:
        # Standard failure check
        logging.error(f"[{instance_id}] Failed to clone repository. Ret: {retcode_c}, Timed Out: {timed_out_c}\nStderr: {stderr_c}")
        return None, f"CLONE_FAILED (Ret: {retcode_c}, Timeout: {timed_out_c})"

    # Check stderr even on success, as git might print warnings
    if stderr_c and stderr_c.strip():
         # Log warnings from git clone stderr (e.g., detached HEAD warnings, etc.)
         logging.warning(f"[{instance_id}] Git clone completed with warnings/output on stderr:\n{stderr_c}")

    # Add a small delay - very unlikely to be needed, but cheap to add
    time.sleep(0.5)

    # Explicitly check if the target directory *and* the .git directory exist NOW
    dot_git_path = os.path.join(instance_workspace, '.git')
    if not os.path.exists(instance_workspace): # Should exist since we created it
         logging.error(f"[{instance_id}] CRITICAL: Target workspace directory '{instance_workspace}' missing after clone attempt!")
         return None, "CLONE_VERIFY_FAILED (Workspace dir missing)"
    if not os.path.exists(dot_git_path) or not os.path.isdir(dot_git_path):
         logging.error(f"[{instance_id}] CRITICAL: .git directory NOT FOUND at '{dot_git_path}' after supposedly successful clone!")
         # Log directory contents again for clues
         try:
             contents = os.listdir(instance_workspace)
             logging.error(f"[{instance_id}] Contents of '{instance_workspace}' after clone attempt: {contents}")
         except Exception as list_err:
             logging.error(f"[{instance_id}] Could not list contents of '{instance_workspace}' after clone attempt: {list_err}")
         return None, "CLONE_VERIFY_FAILED (.git dir missing)"
    # --- End Enhanced Check ---

    logging.info(f"[{instance_id}] Clone appears successful. Proceeding to checkout.")

    # Checkout specific commit (remains the same)
    logging.info(f"[{instance_id}] Checking out commit {base_commit}...")
    cmd_checkout = ["git", "checkout", base_commit]
    # Run checkout *inside* the instance_workspace
    stdout_co, stderr_co, retcode_co, timed_out_co = run_command(cmd_checkout, instance_workspace, timeout=60)
    if retcode_co != 0 or timed_out_co:
        logging.error(f"[{instance_id}] Failed to checkout commit {base_commit}. Ret: {retcode_co}, Timed Out: {timed_out_co}\nStderr: {stderr_co}")
        return None, f"CHECKOUT_FAILED (Ret: {retcode_co}, Timeout: {timed_out_co})"

    logging.info(f"[{instance_id}] Workspace setup complete.")
    return instance_workspace, None # Return path and no error message

# Add this function to your run_java_tests.py script

def parse_search_replace_block(block_string):
    """
    Parses a single SEARCH/REPLACE block string into its components.

    Args:
        block_string (str): The raw string for one SEARCH/REPLACE block.

    Returns:
        tuple: (file_path, search_content, replace_content, error_message)
               Returns None for content parts if parsing fails, along with an error message.
    """
    lines = block_string.strip().split('\n')
    if not lines or not lines[0].startswith("### "):
        return None, None, None, "Invalid block header (missing '### ')"

    file_path = lines[0][4:].strip() # Get path after "### "

    try:
        # Find marker indices
        search_start_index = -1
        separator_index = -1
        replace_end_index = -1

        for i, line in enumerate(lines):
            if line == "<<<<<<< SEARCH":
                search_start_index = i
            elif line == "=======":
                separator_index = i
            elif line == ">>>>>>> REPLACE":
                replace_end_index = i
                break # Found end marker

        # Validate marker presence and order
        if not (0 <= search_start_index < separator_index < replace_end_index):
             markers_found = f"SEARCH={search_start_index}, SEP={separator_index}, END={replace_end_index}"
             return file_path, None, None, f"Malformed block structure (markers missing or out of order: {markers_found})"

        # Extract content, preserving original line endings by re-joining.
        # Add back the newline characters that split('\n') removed.
        # Slice notation [start+1 : end] excludes the markers themselves.

        if search_start_index + 1 == separator_index:
            # Handle empty search block (e.g., for file additions)
            search_content = ""
        else:
            search_content = "\n".join(lines[search_start_index + 1 : separator_index]) + "\n"

        if separator_index + 1 == replace_end_index:
            # Handle empty replace block (e.g., for file deletions)
            replace_content = ""
        else:
            replace_content = "\n".join(lines[separator_index + 1 : replace_end_index]) + "\n"

        # Remove potentially doubled final newline if the last content line didn't have one
        # (Though usually they do. Let's refine if needed)
        # Example refinement:
        # if lines[separator_index-1] == '' and search_content.endswith('\n\n'): search_content = search_content[:-1]
        # if lines[replace_end_index-1] == '' and replace_content.endswith('\n\n'): replace_content = replace_content[:-1]


        return file_path, search_content, replace_content, None # No error

    except Exception as e:
        logging.error(f"Error parsing block for {file_path}: {e}", exc_info=True)
        return file_path, None, None, f"Unexpected error parsing block: {e}"


def apply_search_replace(instance_workspace_dir, search_replace_blocks):
    """
    Applies a list of SEARCH/REPLACE blocks to files within a workspace.
    Processes changes file by file, applying blocks sequentially in memory.

    Args:
        instance_workspace_dir (str): The root directory of the checked-out code.
        search_replace_blocks (list): A list of SEARCH/REPLACE block strings.

    Returns:
        str: None if successful, otherwise an error message describing the failure.
    """
    # Group blocks by the target file path they affect
    blocks_by_file = {}
    block_parse_errors = []
    for i, block_str in enumerate(search_replace_blocks):
        # Skip empty strings or error markers from conversion step
        if not block_str or block_str.startswith("PATCH_PARSE_ERROR"):
            logging.warning(f"Skipping invalid or error block #{i+1}: {block_str[:100]}...")
            continue

        # Parse the block to get the file path
        file_path, _, _, error = parse_search_replace_block(block_str)
        if error:
            logging.error(f"Failed to parse block #{i+1} targeting '{file_path}': {error}")
            # Collect parse errors but continue grouping others
            block_parse_errors.append(f"Block #{i+1} (target: {file_path}): {error}")
            continue # Skip this block for application

        if file_path not in blocks_by_file:
            blocks_by_file[file_path] = []
        blocks_by_file[file_path].append(block_str) # Store the raw block string

    if block_parse_errors:
        return f"Encountered errors parsing blocks: {'; '.join(block_parse_errors)}"

    # Apply changes file by file
    for relative_file_path, blocks in blocks_by_file.items():
        absolute_file_path = os.path.normpath(os.path.join(instance_workspace_dir, relative_file_path))
        logging.info(f"Processing {len(blocks)} block(s) for: {relative_file_path}")

        # --- Check if file exists ---
        file_exists = os.path.exists(absolute_file_path)
        # Peek at first block to see if it's potentially an addition (empty search)
        _, first_search_content, _, _ = parse_search_replace_block(blocks[0])
        is_addition = (first_search_content == "")

        if not file_exists and not is_addition:
            msg = f"Patch Error: Target file does not exist and block is not an addition: {absolute_file_path}"
            logging.error(msg)
            return msg
        elif file_exists and is_addition:
            logging.warning(f"File '{relative_file_path}' exists but first block has empty SEARCH. Assuming overwrite/replacement of entire content.")
            # Treat as starting with existing content, which will be replaced by first block


        try:
            # --- Read original content ---
            current_content = ""
            if file_exists:
                # Try reading with utf-8, fallback if needed
                try:
                    with open(absolute_file_path, 'r', encoding='utf-8') as f:
                        current_content = f.read()
                except UnicodeDecodeError:
                    logging.warning(f"UTF-8 decode failed for {relative_file_path}. Trying latin-1.")
                    with open(absolute_file_path, 'r', encoding='latin-1') as f:
                        current_content = f.read()

            original_content_for_file = current_content # Keep for error logging if needed

            # --- Apply blocks sequentially to the content in memory ---
            block_num = 0
            for block_str in blocks:
                block_num += 1
                file_path, search_content, replace_content, error = parse_search_replace_block(block_str)
                # Error check already done above, but good practice
                if error: return f"Internal Error: Re-parsing failed for block {block_num} of {relative_file_path}"

                # Handle different block types
                if search_content == "" and replace_content == "":
                    logging.warning(f"Skipping empty SEARCH and REPLACE block #{block_num} for {relative_file_path}")
                    continue
                elif search_content == "": # File Addition / Overwrite
                    if block_num > 1 and not file_exists:
                         # This is odd: adding content to a non-existent file after the first block?
                         return f"Patch Error: Block #{block_num} for {relative_file_path} is ADD block but file should have been created by block #1."
                    logging.debug(f"Applying ADD/OVERWRITE block #{block_num} to {relative_file_path}")
                    current_content = replace_content
                    file_exists = True # Mark as existing now
                else: # Standard Search and Replace (or File Deletion if replace_content is "")
                     if not file_exists:
                          return f"Patch Error: Trying to apply SEARCH block #{block_num} to non-existent file {relative_file_path}"

                     # Use string.find() to locate the exact search block
                     start_index = current_content.find(search_content)

                     if start_index == -1:
                         # Search content not found! Patch doesn't apply cleanly.
                         msg = f"Patch Error: SEARCH block #{block_num} not found in {relative_file_path}.\n------ SEARCH BLOCK ------\n{search_content}\n------ END SEARCH ------"
                         logging.error(msg)
                         # Optionally log snippets of current_content for debugging
                         # logging.debug(f"Current content excerpt: {current_content[max(0,start_index-50):start_index+50]}")
                         return msg # Fail fast

                     # Perform replacement
                     logging.debug(f"Applying REPLACE block #{block_num} to {relative_file_path}")
                     # Slice the string to replace only the first occurrence
                     current_content = current_content[:start_index] + replace_content + current_content[start_index + len(search_content):]

            # --- Write the fully modified content back to the file ---
            # Ensure parent directory exists before writing (important for added files)
            parent_dir = os.path.dirname(absolute_file_path)
            if not os.path.exists(parent_dir):
                try:
                    os.makedirs(parent_dir)
                    logging.info(f"Created directory: {parent_dir}")
                except OSError as e:
                    msg = f"Failed to create directory {parent_dir} for {relative_file_path}: {e}"
                    logging.error(msg)
                    return msg

            try:
                with open(absolute_file_path, 'w', encoding='utf-8') as f:
                    f.write(current_content)
            except Exception as e_write:
                 # Try writing with latin-1 as a fallback? Maybe not ideal.
                 msg = f"Failed to write modified content to {absolute_file_path}: {e_write}"
                 logging.error(msg)
                 return msg

        except IOError as e:
             msg = f"File I/O error processing {absolute_file_path}: {e}"
             logging.error(msg)
             return msg
        except Exception as e:
             msg = f"Unexpected error applying blocks to {absolute_file_path}: {e}"
             logging.error(msg, exc_info=True)
             return msg

    logging.info("Finished applying SEARCH/REPLACE blocks for this instance.")
    return None # Indicate success

# Add these functions to your run_java_tests.py script
# Make sure imports like os, subprocess, shutil, logging, tempfile are present

def apply_diff_patch(patch_content, repo_dir, instance_id):
    """
    Applies a standard git diff patch, typically for adding the test file.

    Args:
        patch_content (str): The content of the Git diff patch.
        repo_dir (str): The path to the repository workspace.
        instance_id (str): For logging purposes.

    Returns:
        str: None if successful, otherwise an error message.
    """
    if not patch_content or not patch_content.strip():
        logging.warning(f"[{instance_id}] Provided test patch content is empty. Skipping application.")
        # This might be okay if the test generation failed for this instance
        return None # Not necessarily an error, just nothing to apply

    # Use a temporary file to store the patch content
    # 'tempfile.NamedTemporaryFile' creates a file that is deleted on close
    # 'delete=False' keeps it until manually removed, useful for debugging sometimes
    # Ensure text mode and utf-8 encoding
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".diff", encoding='utf-8') as temp_patch_file:
            temp_patch_file.write(patch_content)
            temp_patch_path = temp_patch_file.name
        logging.info(f"[{instance_id}] Applying test patch from temporary file: {temp_patch_path}")

        # Use git apply to apply the patch
        # --index: Stores the patch in the index (staging area)
        # --unsafe-paths: Allows patching files outside the current directory (shouldn't be needed here but sometimes helps)
        # --reject: Leave rejected hunks in .rej files instead of failing outright (helps debugging)
        # --whitespace=fix: Tries to fix whitespace errors automatically
        # Check first? 'git apply --check' can be used, but let's apply directly.
        cmd_apply = ["git", "apply", "--whitespace=fix", temp_patch_path]
        stdout, stderr, retcode, timed_out = run_command(cmd_apply, repo_dir, timeout=60)

        if retcode != 0:
            error_msg = f"PATCH_APPLY_FAILED (Test Patch): `git apply` failed with code {retcode}.\nStderr: {stderr}"
            logging.error(f"[{instance_id}] {error_msg}")
            # Log stdout too, it might contain useful info from git apply
            logging.error(f"[{instance_id}] Stdout: {stdout}")
            return error_msg
        else:
            logging.info(f"[{instance_id}] Successfully applied test patch.")
            return None # Success

    except Exception as e:
        error_msg = f"PATCH_APPLY_EXCEPTION (Test Patch): Unexpected error applying patch: {e}"
        logging.error(f"[{instance_id}] {error_msg}", exc_info=True)
        return error_msg
    finally:
        # Clean up the temporary file
        if 'temp_patch_path' in locals() and os.path.exists(temp_patch_path):
            try:
                os.remove(temp_patch_path)
            except OSError as e_clean:
                logging.warning(f"[{instance_id}] Failed to remove temporary patch file {temp_patch_path}: {e_clean}")


def run_maven_test(repo_dir, test_class_name, timeout, instance_id, target_module=None):
    """
    Runs a specific test class using Maven. First attempts to compile,
    then runs the test if compilation succeeds. Includes -U flag to force
    snapshot updates and attempts to override jacocoArgLine.
    Optionally targets a module.

    Args:
        repo_dir (str): Path to the repository workspace.
        test_class_name (str): The simple name of the test class (e.g., "ReproduceBugTest").
        timeout (int): Timeout in seconds for the *entire* Maven process (compile + test).
        instance_id (str): For logging purposes.
        target_module (str, optional): The name of the module to run tests in. Defaults to None.

    Returns:
        tuple: (stdout, stderr, return_code, timeout_expired) - Represents the final test execution result.
    """
    # Basic check for pom.xml at the root
    if not os.path.exists(os.path.join(repo_dir, "pom.xml")):
        logging.error(f"[{instance_id}] No pom.xml found in {repo_dir}. Cannot run Maven.")
        return "", "No pom.xml found", -3, False # Custom error code

    # --- Step 1: Attempt Compilation (including test sources) ---
    logging.info(f"[{instance_id}] Attempting Maven compilation (including test sources)...")
    cmd_compile = [
        "mvn", "-B", "-ntp",
        "-U", # <<< --- ADDED: Force update of snapshots ---
        "clean",
        "test-compile", # Compile main and test sources
        "-e" # Show detailed errors if compilation fails
    ]
    if target_module:
        module_arg = f":{target_module}"
        cmd_compile.extend(["-pl", module_arg, "-am"])
        logging.info(f"[{instance_id}] Targeting module {module_arg} for compilation.")

    compile_timeout = timeout / 2 if timeout else None
    stdout_compile, stderr_compile, retcode_compile, timed_out_compile = run_command(
        cmd_compile, repo_dir, timeout=compile_timeout
    )

    if timed_out_compile:
        logging.error(f"[{instance_id}] Maven compilation timed out after {compile_timeout}s.")
        return stdout_compile, stderr_compile, -1, True # Report timeout
    if retcode_compile != 0:
        # Check if it's the POM resolution error before logging generic failure
        if "Non-resolvable parent POM" in stdout_compile or "Could not resolve dependencies" in stdout_compile:
             logging.error(f"[{instance_id}] Maven build failed due to dependency resolution error (likely missing SNAPSHOT). RC: {retcode_compile}")
             # Log specific snippets if helpful
             logging.error(f"[{instance_id}] Relevant compilation stdout snippet:\n{stdout_compile[-1500:]}") # Log end of stdout
        else:
             logging.error(f"[{instance_id}] Maven compilation failed with return code {retcode_compile}.")
             logging.error(f"[{instance_id}] Compilation stdout:\n{stdout_compile}")
             logging.error(f"[{instance_id}] Compilation stderr:\n{stderr_compile}")
        return stdout_compile, stderr_compile, retcode_compile, False # Return specific failure

    logging.info(f"[{instance_id}] Maven compilation successful. Proceeding to test execution.")

    # --- Step 2: Run Surefire Tests (if compilation succeeded) ---
    test_pattern = f"**/{test_class_name}"
    logging.info(f"[{instance_id}] Using test pattern for Surefire: {test_pattern}")

    cmd_test = [
        "mvn", "-B", "-ntp",
        "-U", # <<< --- ADDED: Force update of snapshots ---
        "org.apache.maven.plugins:maven-surefire-plugin:3.2.5:test", # Use specific goal
        f"-Dtest={test_pattern}",
        "-DfailIfNoTests=false",
        "-DjacocoArgLine= "  # Attempt to override/blank the jacoco arg
    ]

    if target_module:
        module_arg = f":{target_module}"
        cmd_test.extend(["-pl", module_arg]) # Target the specific module for test execution
        logging.info(f"[{instance_id}] Targeting module {module_arg} for test execution.")

    logging.info(f"[{instance_id}] Running Maven Surefire test command: {' '.join(cmd_test)}")
    test_timeout = timeout - compile_timeout if timeout and compile_timeout else timeout
    stdout_test, stderr_test, retcode_test, timed_out_test = run_command(
        cmd_test, repo_dir, timeout=test_timeout
    )

    if timed_out_test:
        logging.warning(f"[{instance_id}] Maven test execution timed out after {test_timeout}s.")
    elif retcode_test != 0:
        logging.warning(f"[{instance_id}] Maven Surefire execution failed with return code {retcode_test}.")
    else:
        logging.info(f"[{instance_id}] Maven Surefire execution completed successfully (Return Code 0).")

    # Combine outputs for analysis function (optional, could just pass test outputs)
    combined_stdout = f"--- Compile Output ---\n{stdout_compile}\n--- Test Output ---\n{stdout_test}"
    combined_stderr = f"--- Compile Stderr ---\n{stderr_compile}\n--- Test Stderr ---\n{stderr_test}"

    # Return the results of the test execution phase
    return combined_stdout, combined_stderr, retcode_test, timed_out_test


def analyze_test_output(stdout, stderr, return_code, timeout_expired, instance_id):
    """
    Determines the outcome based on test execution results (stdout/stderr/return code).
    Includes enhanced logging for failures.

    Args:
        stdout (str): Standard output from the test command.
        stderr (str): Standard error from the test command.
        return_code (int): Exit code from the test command.
        timeout_expired (bool): Whether the command timed out.
        instance_id (str): For logging purposes.

    Returns:
        str: A status string like "RESOLVED", "REPRODUCED", "OTHER_ISSUES", "TEST_TIMEOUT", "TEST_ERROR".
    """
    logging.info(f"[{instance_id}] Analyzing test output. Return code: {return_code}, Timed out: {timeout_expired}")

    # Combine stdout and stderr for searching marker strings
    full_output = stdout + "\n" + stderr

    # Check for timeout first
    if timeout_expired:
        logging.warning(f"[{instance_id}] Outcome: TEST_TIMEOUT")
        return "TEST_TIMEOUT"

    # Check for explicit markers in the output (prioritize RESOLVED)
    if "ISSUE_RESOLVED" in full_output:
        logging.info(f"[{instance_id}] Outcome: RESOLVED (found 'ISSUE_RESOLVED')")
        return "RESOLVED"
    elif "ISSUE_REPRODUCED" in full_output:
        logging.info(f"[{instance_id}] Outcome: REPRODUCED (found 'ISSUE_REPRODUCED')")
        return "REPRODUCED"
    elif "OTHER_ISSUES" in full_output:
         logging.warning(f"[{instance_id}] Outcome: OTHER_ISSUES (found 'OTHER_ISSUES')")
         return "OTHER_ISSUES"

    # --- Enhanced Logging for Errors ---
    # If no markers found, check the return code from Maven/Gradle
    if return_code != 0:
        outcome = f"TEST_ERROR (Code: {return_code})"
        logging.warning(f"[{instance_id}] Outcome: {outcome} (No markers found)")
        # Log the detailed output to help diagnose the Maven failure
        logging.warning(f"[{instance_id}] Maven stdout for error diagnosis:\n------ STDOUT START ------\n{stdout}\n------ STDOUT END ------")
        logging.warning(f"[{instance_id}] Maven stderr for error diagnosis:\n------ STDERR START ------\n{stderr}\n------ STDERR END ------")
        return outcome

    # If markers are missing but the command succeeded (return code 0)
    outcome = "TEST_ERROR (No markers, RC=0)"
    logging.warning(f"[{instance_id}] Outcome: {outcome} (Command succeeded but no expected markers found)")
    logging.warning(f"[{instance_id}] Maven stdout for ambiguous diagnosis:\n------ STDOUT START ------\n{stdout}\n------ STDOUT END ------")
    logging.warning(f"[{instance_id}] Maven stderr for ambiguous diagnosis:\n------ STDERR START ------\n{stderr}\n------ STDERR END ------")
    return outcome



def run_single_instance_test(instance_data, args):
    """
    Orchestrates the testing process for one instance, targeting the correct module.
    """
    instance_id = instance_data['instance_id']
    repo_slug = instance_data['repo']
    base_commit = instance_data['base_commit']
    search_replace_blocks = instance_data['search_replace_blocks']
    test_patch_content = instance_data['test_patch'] # Content of the generated test patch

    instance_workspace = None # Initialize to ensure cleanup happens
    final_outcome = "UNKNOWN_ERROR" # Default outcome

    try:
        # 1. Setup Workspace
        logging.info(f"[{instance_id}] Starting test run.")
        instance_workspace, setup_error = setup_workspace(
            instance_id, repo_slug, base_commit, args.workspace_dir, args.run_id
        )
        if setup_error:
            logging.error(f"[{instance_id}] Workspace setup failed: {setup_error}")
            return f"SETUP_FAILED: {setup_error}"

        # 2. Apply Code Patch (SEARCH/REPLACE)
        logging.info(f"[{instance_id}] Applying code patch (SEARCH/REPLACE format)...")
        apply_sr_error = apply_search_replace(instance_workspace, search_replace_blocks)
        if apply_sr_error:
            logging.error(f"[{instance_id}] Failed to apply SEARCH/REPLACE patch: {apply_sr_error}")
            return f"CODE_PATCH_FAILED: {apply_sr_error}"
        logging.info(f"[{instance_id}] Code patch applied successfully.")

        # --- Determine Target Module ---
        target_module = None
        if search_replace_blocks:
            # Parse the first block to get a file path
            first_block_str = next((b for b in search_replace_blocks if b and not b.startswith("PATCH_PARSE_ERROR")), None)
            if first_block_str:
                file_path, _, _, _ = parse_search_replace_block(first_block_str)
                if file_path:
                    # Assume module name is the first directory component
                    path_parts = file_path.split(os.path.sep)
                    if len(path_parts) > 1:
                        target_module = path_parts[0]
                        logging.info(f"[{instance_id}] Inferred target module: {target_module} from path: {file_path}")

        if not target_module:
             logging.warning(f"[{instance_id}] Could not determine target module from code patch paths. Will run test from root.")
             # Proceed without -pl flag, might still fail
        # --- End Determine Target Module ---


        # Check for Empty Test Patch BEFORE Applying/Running
        if not test_patch_content or not test_patch_content.strip():
            logging.warning(f"[{instance_id}] Test patch content is empty. Test generation likely failed.")
            final_outcome = "TEST_GENERATION_FAILED"
            return final_outcome # Exit early for this instance

        # 3. Apply Test Patch (Standard Diff)
        logging.info(f"[{instance_id}] Applying generated test patch...")
        apply_test_patch_error = apply_diff_patch(test_patch_content, instance_workspace, instance_id)
        if apply_test_patch_error:
            logging.error(f"[{instance_id}] Failed to apply test patch: {apply_test_patch_error}")
            return f"TEST_PATCH_FAILED: {apply_test_patch_error}"

        # --- Check where test file landed (optional debug) ---
        test_file_path_check = os.path.join(instance_workspace, "ReproduceBugTest.java")
        if os.path.exists(test_file_path_check):
            logging.info(f"[{instance_id}] Confirmed test file exists at: {test_file_path_check}")
        else:
            # Check common module location as fallback
            if target_module:
                test_file_path_check_module = os.path.join(instance_workspace, target_module, "src", "test", "java", "ReproduceBugTest.java")
                if os.path.exists(test_file_path_check_module):
                     logging.info(f"[{instance_id}] Confirmed test file exists at module path: {test_file_path_check_module}")
                else:
                     logging.warning(f"[{instance_id}] Test file not found at root or expected module path after applying patch.")
            else:
                 logging.warning(f"[{instance_id}] Test file not found at root after applying patch.")
        # --- End Check ---

        logging.info(f"[{instance_id}] Test patch applied successfully.") # Log success even if file check fails

        # 4. Run the Test (Passing target module if found)
        logging.info(f"[{instance_id}] Running Maven test...")
        test_class_name = "ReproduceBugTest" # Name used in the generation prompt
        stdout, stderr, retcode, timed_out = run_maven_test(
            instance_workspace,
            test_class_name,
            args.timeout,
            instance_id,
            target_module=target_module # Pass the inferred module
        )

        # 5. Analyze Test Output
        logging.info(f"[{instance_id}] Analyzing test results...")
        final_outcome = analyze_test_output(stdout, stderr, retcode, timed_out, instance_id)
        logging.info(f"[{instance_id}] Final outcome: {final_outcome}")

        return final_outcome # Return the analyzed result

    except Exception as e:
        logging.error(f"[{instance_id}] Unexpected exception during testing: {e}", exc_info=True)
        final_outcome = f"RUNNER_UNEXPECTED_EXCEPTION: {e}" # Update outcome on exception
        return final_outcome # Return error outcome
    finally:
        # 6. Cleanup Workspace (Optional)
        if args.cleanup_workspace and instance_workspace and os.path.exists(instance_workspace):
            logging.info(f"[{instance_id}] Cleaning up workspace: {instance_workspace}")
            try:
                shutil.rmtree(instance_workspace)
            except OSError as e_clean:
                logging.error(f"[{instance_id}] Failed to cleanup workspace {instance_workspace}: {e_clean}")
        elif instance_workspace:
             logging.info(f"[{instance_id}] Workspace kept at: {instance_workspace}")



def main():
    parser = argparse.ArgumentParser(description="Run generated Java reproduction tests against code patches.")

    # Input Files
    parser.add_argument("--converted_patch_file", type=str, required=True, help="Path to JSONL file with SEARCH/REPLACE code patches (output of convert_diff_format.py).")
    parser.add_argument("--generated_test_file", type=str, required=True, help="Path to JSONL file with generated test patches (output_N... file from generate_reproduction_tests.py).")
    # Dataset Info (Optional if info already merged, but good practice to load fresh)
    parser.add_argument("--dataset_name", type=str, default="Daoguang/Multi-SWE-bench", help="Name of the Hugging Face dataset.")
    parser.add_argument("--dataset_split", type=str, default="java_verified", help="Split of the dataset to use.")
    parser.add_argument("--dataset_cache_dir", type=str, default=None, help="Directory to cache Hugging Face datasets.")

    # Execution Configuration
    parser.add_argument("--workspace_dir", type=str, required=True, help="Base directory to create temporary workspaces for checkouts.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the final validation results (JSONL).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel processes to run tests.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for running the test command (mvn/gradle).")
    parser.add_argument("--run_id", type=str, default=time.strftime("%Y%m%d_%H%M%S"), help="Identifier for this run (used in logging/temp dirs).")
    parser.add_argument("--instance_ids", nargs='+', default=None, help="Optional list of specific instance_ids to run.")
    parser.add_argument("--skip_existing", action='store_true', help="Skip instances already present in the results file.")
    parser.add_argument("--cleanup_workspace", action='store_true', help="Remove instance workspace after testing.")

    args = parser.parse_args()

    # --- Load Data ---
    logging.info("Loading data...")
    try:
        code_patches = {item['instance_id']: item['search_replace_blocks'] for item in load_jsonl(args.converted_patch_file)}
        logging.info(f"Loaded {len(code_patches)} code patches from {args.converted_patch_file}")
    except Exception as e:
        logging.error(f"Failed to load code patches from {args.converted_patch_file}: {e}", exc_info=True)
        return

    try:
        test_patches = {item['instance_id']: item['test_patch'] for item in load_jsonl(args.generated_test_file)}
        logging.info(f"Loaded {len(test_patches)} test patches from {args.generated_test_file}")
    except Exception as e:
        logging.error(f"Failed to load test patches from {args.generated_test_file}: {e}", exc_info=True)
        return

    try:
        logging.info(f"Loading dataset '{args.dataset_name}' split '{args.dataset_split}'...")
        dataset = load_dataset(args.dataset_name, split=args.dataset_split, cache_dir=args.dataset_cache_dir, trust_remote_code=True)
        dataset_info = {item['instance_id']: {'repo': item['repo'], 'base_commit': item['base_commit']} for item in dataset}
        logging.info(f"Loaded info for {len(dataset_info)} instances from dataset.")
    except Exception as e:
        logging.error(f"Failed to load dataset info: {e}", exc_info=True)
        return

    # --- Prepare Instance List ---
    all_instance_ids = set(code_patches.keys()) & set(test_patches.keys()) & set(dataset_info.keys())
    logging.info(f"Found {len(all_instance_ids)} instances common across all input sources.")

    if args.instance_ids:
        target_ids = set(args.instance_ids)
        run_instance_ids = list(all_instance_ids & target_ids)
        logging.info(f"Filtered to {len(run_instance_ids)} instances based on --instance_ids argument.")
    else:
        run_instance_ids = list(all_instance_ids)

    if not run_instance_ids:
        logging.warning("No instances selected to run. Exiting.")
        return

    # --- Skip Existing ---
    existing_results = {}
    if args.skip_existing and os.path.exists(args.results_file):
        logging.info(f"Loading existing results from {args.results_file} to skip...")
        existing_results = {item['instance_id']: item for item in load_jsonl(args.results_file)}
        original_count = len(run_instance_ids)
        run_instance_ids = [id for id in run_instance_ids if id not in existing_results]
        logging.info(f"Skipped {original_count - len(run_instance_ids)} instances found in existing results.")

    if not run_instance_ids:
        logging.info("No new instances to run after skipping. Exiting.")
        return

    # --- Create combined data structure ---
    instances_to_run = []
    for instance_id in run_instance_ids:
        instances_to_run.append({
            'instance_id': instance_id,
            'search_replace_blocks': code_patches[instance_id],
            'test_patch': test_patches[instance_id],
            'repo': dataset_info[instance_id]['repo'],
            'base_commit': dataset_info[instance_id]['base_commit']
        })

    # --- Run Tests ---
    os.makedirs(args.workspace_dir, exist_ok=True)
    results = {}

    logging.info(f"Starting test execution for {len(instances_to_run)} instances using {args.num_workers} worker(s)...")
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit tasks
            future_to_instance = {
                executor.submit(run_single_instance_test, instance_data, args): instance_data['instance_id']
                for instance_data in instances_to_run
            }
            # Process results as they complete
            for future in tqdm(as_completed(future_to_instance), total=len(instances_to_run), desc="Running Tests"):
                instance_id = future_to_instance[future]
                try:
                    outcome = future.result()
                    results[instance_id] = outcome
                except Exception as exc:
                    logging.error(f'{instance_id} generated an exception: {exc}', exc_info=True)
                    results[instance_id] = "RUNNER_EXCEPTION"
    else: # Run sequentially for easier debugging
        logging.info("Running in single-process mode.")
        for instance_data in tqdm(instances_to_run, desc="Running Tests"):
             instance_id = instance_data['instance_id']
             try:
                 outcome = run_single_instance_test(instance_data, args)
                 results[instance_id] = outcome
             except Exception as exc:
                 logging.error(f'{instance_id} generated an exception: {exc}', exc_info=True)
                 results[instance_id] = "RUNNER_EXCEPTION"

    # --- Save Results ---
    logging.info(f"Finished testing. Saving {len(results)} results to {args.results_file}")
    # Append new results to existing ones if skip_existing was used
    final_results_list = list(existing_results.values()) # Start with existing results
    for instance_id, outcome in results.items():
         final_results_list.append({
             "instance_id": instance_id,
             "outcome": outcome,
             "run_id": args.run_id
         })

    try:
        with open(args.results_file, 'w', encoding='utf-8') as f:
            for entry in final_results_list:
                f.write(json.dumps(entry) + '\n')
        logging.info("Results saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save results: {e}", exc_info=True)

    logging.info("Script finished.")


if __name__ == "__main__":
    # Need tqdm for progress bars
    from tqdm import tqdm
    main()