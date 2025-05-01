# generate_reproduction_tests.py
import argparse
import json
import os
import re # Make sure re is imported
from collections import Counter
from threading import Lock
import logging
import time
import google.generativeai as genai
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
import pathlib
import concurrent.futures # Ensure this is imported

# --- Utility Functions ---

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

def setup_logger(log_file):
    """Sets up a logger for a specific file."""
    logger = logging.getLogger(log_file) # Use filename as logger name
    logger.setLevel(logging.INFO)
    # Prevent adding multiple handlers if called repeatedly
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger

# --- LLM Interaction ---

# Prompt template with escaped braces for .format()
# Refined prompt template for generate_reproduction_tests.py

generate_java_tests_prompt_template = """
You are an expert Java test engineer. You are tasked with generating a Java test case to reproduce a specific issue described in a bug report. Your generated code MUST be syntactically correct and compile without errors according to standard Java rules.

Here is the issue description:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Please generate a *complete, runnable* Java test class using JUnit 5 that demonstrates this issue.

The test class MUST adhere to the following requirements:
1.  Be named `ReproduceBugTest`.
2.  **DO NOT include a `package` declaration.** The class should be in the default package.
3.  Contain necessary Java imports (e.g., `org.junit.jupiter.api.Test`, `static org.junit.jupiter.api.Assertions.*`, relevant classes from the issue). Ensure all types used are imported.
4.  Include one or more `@Test` methods that reproduce the scenario described in the issue.
5.  Inside the test method(s), execute the code that triggers the bug.
6.  **Pay close attention to type correctness, especially when using Java Reflection (`java.lang.reflect.*`) or Generics.** Ensure variables are declared with the exact type returned by methods (e.g., `java.lang.reflect.Type[]` vs `Class<?>[]`). Use explicit type casting only if necessary and safe.
7.  Use JUnit assertions (`assertEquals`, `assertThrows`, `assertTrue`, etc.) to check the outcome. The assertion should ideally *fail* if the bug is present.
8.  If the assertion passes *unexpectedly* (meaning the bug is *not* present or already fixed), print the exact string "ISSUE_RESOLVED" to standard output.
9.  If the assertion fails as *expected* (meaning the bug *is* present), print the exact string "ISSUE_REPRODUCED" to standard output. Catch the specific `AssertionError` or the expected exception that indicates the bug.
10. If any other unexpected exception occurs during test execution (e.g., `NullPointerException`, `ClassCastException` when not expected by the bug itself, compilation errors indication), print the exact string "OTHER_ISSUES" to standard output. Use a general `catch (Exception e)` block for this *after* catching specific expected exceptions or `AssertionError`.
11. Ensure only ONE of the strings "ISSUE_RESOLVED", "ISSUE_REPRODUCED", or "OTHER_ISSUES" is printed per test execution path.
12. **If you define any helper interfaces or classes inside the `ReproduceBugTest.java` file for test setup, they MUST NOT be declared `public`.** They should have default (package-private) access or be inner classes.

Example Structure (No package declaration, no public helpers):

```java
// NO package declaration

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
// Add other necessary imports based on the issue context, e.g.:
// import java.lang.reflect.Method;
// import java.lang.reflect.Type;

// --- Helper interface with default access (NOT public) ---
interface ExampleInterface {{ // Escaped brace
    String helperMethod();
}} // Escaped brace
// --- End Helper ---


public class ReproduceBugTest {{ // Escaped brace

    // Inner classes are also allowed
    private static class InnerHelper {{ // Escaped brace
        // ...
    }} // Escaped brace


    @Test
    void testIssueScenario() {{ // Escaped brace
        try {{ // Escaped brace
            // 1. Setup code related to the issue (if needed)
            // Example: Method method = SomeClass.class.getMethod("buggyMethod");
            ExampleInterface helper = () -> "test"; // Using helper

            // 2. Code that is expected to trigger the bug
            // Example: Type[] result = SomeReflectionUtil.getTypes(method); // Use correct type

            // 3. Assertion that *should fail* if the bug exists
            // Example: assertEquals(1, result.length);

            // 4. If the assertion above PASSED (unexpectedly), the bug is not present/fixed
            System.out.println("ISSUE_RESOLVED");

        }} catch (AssertionError e) {{ // Escaped brace
            // 5. If the assertion FAILED as expected, the bug is reproduced
            System.out.println("ISSUE_REPRODUCED");
        }} catch (NoSuchMethodException e) {{ // Escaped brace
             // Example of catching a specific exception that might occur during setup
             System.err.println("Test setup failed: " + e.getMessage());
             System.out.println("OTHER_ISSUES");
        }} catch (Exception e) {{ // Escaped brace
            // 6. Handle any other unexpected exception during the test
            System.err.println("Test failed with unexpected exception: " + e.getMessage());
            e.printStackTrace(); // Optional: print stack trace for debugging
            System.out.println("OTHER_ISSUES");
        }}
    }} // Escaped brace

    // Add more @Test methods if needed for complex issues
}} // Escaped brace
```

Generate only the complete Java code for the `ReproduceBugTest.java` file.
Wrap the entire Java code block within ```java ... ```.
"""

def call_gemini_api(prompt, model_name, api_key, temperature, max_output_tokens, logger):
    """Calls the Gemini API and returns the response text and token counts."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        # Configure safety settings as needed
        safety_settings = [
             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        if not response.parts:
             logger.warning("Gemini response was empty or blocked.")
             text_response = ""
        else:
             text_response = response.text
        # Placeholder for token counts - replace if API provides usage metadata
        prompt_tokens = 0
        completion_tokens = 0
        return {"response": text_response, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return {"response": f"API_ERROR: {e}", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

# --- Test Generation Core Logic ---

def gen_test(instance_id, args, all_issue_data, prev_o, write_lock=None):
    """Generates a Java test for a given instance_id using Gemini."""
    if args.target_id is not None and args.target_id != instance_id:
        return

    log_file = os.path.join(args.output_folder, "generating_test_logs", f"{instance_id}.log")
    logger = setup_logger(log_file)

    found = any(o["instance_id"] == instance_id for o in prev_o)
    if found and not args.force_regenerate:
        logger.info(f"Skipping {instance_id}: already found in output file {args.output_file}")
        return None

    logger.info(f"================ generating test for {instance_id} ================")

    bench_data = all_issue_data.get(instance_id)
    if not bench_data:
        logger.error(f"Instance ID {instance_id} not found in the loaded dataset dictionary.")
        return None
    problem_statement = bench_data.get("problem_statement", "")
    if not problem_statement:
         logger.error(f"Problem statement missing for instance ID {instance_id}")
         return None

    # Prepare the prompt
    try:
        prompt = generate_java_tests_prompt_template.format(
            problem_statement=problem_statement,
        ).strip()
    except ValueError as e:
         logger.error(f"Failed to format prompt for {instance_id} (check template braces): {e}")
         return None # Cannot proceed without a valid prompt
    except KeyError as e:
         logger.error(f"Missing key '{e}' in prompt template for {instance_id}")
         return None

    logger.info(f"Prompting Gemini ({args.model}) with:\n{prompt[:500]}...")

    # Load API Key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found.")
        error_result = {"instance_id": instance_id, "error": "Missing API Key", "raw_output": [], "all_generations": [], "try_count": [], "traj": []}
        if write_lock: write_lock.acquire()
        try:
            with open(args.output_file, "a", encoding='utf-8') as f: f.write(json.dumps(error_result) + "\n")
        finally:
             if write_lock: write_lock.release()
        return None

    # --- Gemini API Calls ---
    all_generations = []
    raw_outputs = []
    counts = []
    traj = []
    current_try = 0
    while current_try < args.max_samples:
        logger.info(f"Attempting sample {current_try + 1}/{args.max_samples}...")
        api_result = call_gemini_api(
            prompt=prompt, model_name=args.model, api_key=api_key,
            temperature=args.temperature, max_output_tokens=args.max_tokens, logger=logger
        )
        raw_output = api_result["response"]
        if "API_ERROR" in raw_output: logger.error(f"API Error encountered: {raw_output}")
        elif not raw_output.strip(): logger.warning("Received empty response from Gemini.")
        else: logger.info(f"Raw output received (sample {current_try + 1}):\n{raw_output[:500]}...")
        all_generations.append(raw_output)
        raw_outputs.append(raw_output)
        counts.append(current_try + 1)
        traj.append({**api_result, "prompt": prompt})
        current_try += 1
        if args.max_samples > 1 and current_try < args.max_samples: time.sleep(1)

    # --- Write results to output file ---
    output_data = {
        "instance_id": instance_id, "raw_output": raw_outputs,
        "all_generations": [all_generations], "try_count": counts, "traj": traj,
        "prev_content": [[""]], # Keep structure consistent
        "file_names": [["Determine_In_PostProcess"]], # Placeholder, actual path determined later
    }
    if write_lock: write_lock.acquire()
    try:
        with open(args.output_file, "a", encoding='utf-8') as f: f.write(json.dumps(output_data) + "\n")
        logger.info(f"Successfully wrote raw results for {instance_id}")
    except Exception as e:
         logger.error(f"Failed to write raw results for {instance_id}: {e}")
    finally:
        if write_lock: write_lock.release()

    return output_data


def generate_tests(args):
    """Loads filtered IDs, loads corresponding data from Hugging Face dataset, sets up threading, and calls gen_test."""
    os.makedirs(args.output_folder, exist_ok=True)
    # Save arguments used for this run
    with open(os.path.join(args.output_folder, "args_generate.json"), "w", encoding='utf-8') as f:
        args_dict = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}
        json.dump(args_dict, f, indent=4)

    # --- Load filtered patches to get target instance IDs ---
    print(f"Loading filtered instance IDs from: {args.filtered_patches_file}")
    try:
        filtered_patches_data = load_jsonl(args.filtered_patches_file)
        if not filtered_patches_data:
             print(f"Error: No data found or failed to load from {args.filtered_patches_file}. Exiting.")
             return
        target_instance_ids = set(item['instance_id'] for item in filtered_patches_data if 'instance_id' in item)
        if not target_instance_ids:
             print(f"Error: No instance_ids found in {args.filtered_patches_file}. Exiting.")
             return
        print(f"Found {len(target_instance_ids)} unique target instance IDs.")
    except Exception as e:
        print(f"Error loading or processing {args.filtered_patches_file}: {e}. Exiting.")
        return

    # --- Load dataset from Hugging Face to get problem statements ---
    print(f"Loading dataset '{args.dataset_name}' split '{args.dataset_split}' for problem statements...")
    try:
        # Load the full dataset initially
        loaded_dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
        print(f"Loaded dataset with {len(loaded_dataset)} total instances.")
    except Exception as e:
        print(f"Error loading dataset '{args.dataset_name}': {e}. Exiting.")
        return

    # --- Filter dataset and create dictionary keyed by instance_id ---
    print("Filtering dataset and building dictionary for target instances...")
    all_issue_data = {}
    processed_count = 0
    for item in tqdm(loaded_dataset, desc="Filtering dataset"):
        instance_id = item.get("instance_id")
        if instance_id in target_instance_ids:
            # Check if problem_statement exists, log if not but still include
            if "problem_statement" not in item or not item["problem_statement"]:
                 logging.warning(f"Problem statement missing or empty for target instance ID {instance_id} in dataset.")
            all_issue_data[instance_id] = item
            processed_count += 1
            # Optional: Stop early if all target IDs are found
            # if len(all_issue_data) == len(target_instance_ids):
            #    break

    instance_ids_to_process = list(all_issue_data.keys())
    print(f"Dataset filtered. Found data for {len(instance_ids_to_process)}/{len(target_instance_ids)} target instance IDs.")

    if not instance_ids_to_process:
        print("Error: Could not find data for any target instance ID in the dataset. Exiting.")
        return

    # Load previously generated outputs to allow skipping/resuming
    # Note: args.output_file is now set in main() to the raw output file path
    prev_outputs = load_jsonl(args.output_file)
    print(f"Found {len(prev_outputs)} previously processed instances in {args.output_file}")

    # --- Threading and execution ---
    if args.num_threads == 1:
        print("Running in single-threaded mode.")
        for instance_id in tqdm(instance_ids_to_process, desc="Generating tests", colour="cyan"):
            gen_test(instance_id, args, all_issue_data, prev_outputs)
    else:
        print(f"Running with {args.num_threads} threads.")
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {
                # Pass the filtered all_issue_data
                executor.submit(gen_test, instance_id, args, all_issue_data, prev_outputs, write_lock): instance_id
                for instance_id in instance_ids_to_process # Use the filtered list
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(instance_ids_to_process), desc="Generating tests", colour="cyan"):
                instance_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nInstance {instance_id} generated an exception in thread: {exc}")
                    # Setup logger for the main error log, not instance specific here
                    error_log_path = os.path.join(args.output_folder, "error.log")
                    logger = setup_logger(error_log_path) # Use a general error log
                    logger.error(f"Exception during threaded execution for instance {instance_id}: {exc}", exc_info=True)

# --- Post-Processing Logic ---

def extract_first_java_code_block(text):
    """Extracts the first code block enclosed in ```java ... ``` or ``` ... ```"""
    # Try specific ```java block first, allowing whitespace around tags/keyword
    pattern_java = re.compile(r"```\s*java\s*\n?(.*?)```", re.DOTALL)
    match_java = pattern_java.search(text)
    if match_java:
        logging.debug("Matched specific ```java block.") # Use logging
        return match_java.group(1).strip()

    # Fallback: Try generic ``` block, allowing whitespace
    pattern_generic = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)
    match_generic = pattern_generic.search(text)
    if match_generic:
        code = match_generic.group(1).strip()
        # Basic check if it looks like Java
        if "public class" in code or "import " in code or "@Test" in code:
             logging.debug("Matched generic ``` block, looks like Java.") # Use logging
             return code
        else:
             logging.debug(f"Matched generic ``` block, but content doesn't look like Java:\n{code[:100]}...") # Use logging

    logging.warning("Could not extract Java code block using regex.") # Use logging
    return None

def create_patch_from_java_code(java_code: str, full_relative_path_git: str) -> str:
    """Creates a Git diff patch string for adding the given Java code as a new file
       at the specified relative path.
    """
    patch_header = f"""diff --git a/{full_relative_path_git} b/{full_relative_path_git}
new file mode 100644
index 0000000..e69de29
"""
    patch_body = []
    patch_body.append("--- /dev/null")
    patch_body.append(f"+++ b/{full_relative_path_git}")

    code_lines = java_code.strip().split("\n")
    patch_body.append(f"@@ -0,0 +1,{len(code_lines)} @@")

    for line in code_lines:
        patch_body.append(f"+{line}")

    patch_str = patch_header + "\n".join(patch_body)
    if not patch_str.endswith("\n"):
         patch_str += "\n"
    return patch_str

def extract_package_from_java(java_code):
    """Extracts the package declaration from Java code using regex."""
    match = re.search(r"^\s*package\s+([\w\.]+)\s*;", java_code, re.MULTILINE)
    if match:
        package_name = match.group(1)
        package_path = package_name.replace('.', os.path.sep) # Use os.sep for local path part
        logging.debug(f"Extracted package: {package_name} -> path: {package_path}") # Use logging
        return package_path
    else:
        logging.debug("No package declaration found.") # Use logging
        return None

def post_process_tests(args):
    """
    Reads raw outputs, extracts Java code, determines module/package
    using path info from the filtered_patches file, creates patches, and saves them.
    """
    raw_output_file = args.raw_output_file
    # args.output_file (processed output) and args.select_id are set in main() before calling
    processed_output_file = args.output_file
    generation_index_to_select = args.select_id
    filtered_patches_filepath = args.filtered_patches_file # Get path from args

    print(f"Post-processing raw outputs from: {raw_output_file}")
    print(f"Using patch file paths from: {filtered_patches_filepath}")
    print(f"Selecting generation index: {generation_index_to_select}")
    print(f"Writing processed patches to: {processed_output_file}")

    # --- Load filtered patches to get file path info ---
    original_code_patch_info = {}
    try:
        logging.info(f"Loading filtered patches file '{filtered_patches_filepath}' for file path info...")
        filtered_patches_data = load_jsonl(filtered_patches_filepath)
        if not filtered_patches_data:
             logging.error(f"Failed to load or empty data in {filtered_patches_filepath}. Cannot determine test paths.")
             print(f"Error: Failed to load or empty data in {filtered_patches_filepath}. Cannot determine test paths.")
             return # Cannot proceed without path info

        patch_field = 'model_patch' # Use the patch field from the filtered file
        for item in filtered_patches_data:
            instance_id = item.get('instance_id')
            if not instance_id:
                logging.warning("Skipping entry in filtered patches file due to missing 'instance_id'.")
                continue

            # Store the path info only once per instance_id (use the first one found)
            if instance_id not in original_code_patch_info:
                diff_text = item.get(patch_field, "")
                first_file_path = None
                if diff_text:
                    # Extract path from 'diff --git a/path/to/file b/path/to/file'
                    diff_lines = diff_text.split('\n', 1) # Split only the first line
                    if len(diff_lines) > 0 and diff_lines[0].startswith('diff --git a/'):
                        # Split carefully: diff --git a/some/path b/some/other/path
                        parts = diff_lines[0].split(' ')
                        if len(parts) >= 4 and parts[2].startswith('a/'):
                             first_file_path = parts[2][2:] # Remove the 'a/' prefix
                        else:
                             logging.warning(f"Could not parse 'diff --git' line format for {instance_id} in {filtered_patches_filepath}: {diff_lines[0]}")

                if first_file_path:
                    original_code_patch_info[instance_id] = first_file_path
                    logging.debug(f"Stored path '{first_file_path}' for {instance_id}")
                else:
                    # Log only once if path extraction fails for an instance
                    logging.warning(f"Could not extract first file path from '{patch_field}' for {instance_id} in {filtered_patches_filepath}. Test placement might be at root.")

        logging.info(f"Loaded patch file path info for {len(original_code_patch_info)} unique instances.")

    except Exception as e:
        logging.error(f"Failed to load or process {filtered_patches_filepath} for path info: {e}. Cannot reliably place tests.", exc_info=True)
        print(f"Error: Failed to load or process {filtered_patches_filepath}. Cannot reliably place tests.")
        return

    # --- Process Raw Outputs ---
    # Clear output file before writing new results for this index
    try:
        open(processed_output_file, 'w').close()
        logging.info(f"Cleared/created output file: {processed_output_file}")
    except IOError as e:
        logging.error(f"Error clearing output file {processed_output_file}: {e}")
        print(f"Error: Could not clear output file {processed_output_file}. Check permissions.")
        return


    # Load the raw generations produced by generate_tests
    if not os.path.exists(raw_output_file):
         logging.error(f"Raw output file {raw_output_file} not found. Skipping post-processing for index {generation_index_to_select}.")
         print(f"Error: Raw output file {raw_output_file} not found.")
         return
    raw_outputs_data = load_jsonl(raw_output_file)

    processed_count = 0
    error_count = 0
    missing_path_info_count = 0

    for instance_data in tqdm(raw_outputs_data, desc=f"Processing index {generation_index_to_select}", colour="yellow"):
        instance_id = instance_data.get("instance_id")
        if not instance_id:
             logging.warning("Skipping raw output entry with missing instance_id.")
             continue

        # Check if we have path info for this instance_id (handle cases where it might be in raw_outputs but wasn't in filtered_patches)
        original_path = original_code_patch_info.get(instance_id)
        if not original_path:
            logging.warning(f"[{instance_id}] Missing file path info (likely not found in {filtered_patches_filepath}). Test placement might default to root 'repro_test'.")
            # Allow processing to continue, but placement will be default.
            missing_path_info_count += 1 # Track this specific issue

        # Check for generation errors first
        if instance_data.get("error"):
             error_count +=1
             error_entry = { "model_name_or_path": args.model, "instance_id": instance_id, "test_patch": "", "raw_test_patch": f"GENERATION_ERROR: {instance_data['error']}", "original_file_content": "", "error": instance_data['error'] }
             with open(processed_output_file, "a", encoding='utf-8') as f: f.write(json.dumps(error_entry) + "\n")
             continue

        # Get the correct generation based on index
        # Ensure all_generations is treated as list of lists [[gen1_samp1, gen1_samp2],[gen2_samp1]] - adjust based on actual structure
        # Assuming the structure is {"all_generations": [[samp1, samp2, ...]], ...} based on gen_test output
        generations_list = instance_data.get("all_generations", [[]])
        if not generations_list or not isinstance(generations_list[0], list):
             logging.error(f"[{instance_id}] Invalid format for 'all_generations'. Expected list of lists. Found: {generations_list}")
             error_count += 1
             error_entry = { "model_name_or_path": args.model, "instance_id": instance_id, "test_patch": "", "raw_test_patch": "FORMAT_ERROR: Invalid 'all_generations' structure", "original_file_content": "", "error": "Invalid generation data structure" }
             with open(processed_output_file, "a", encoding='utf-8') as f: f.write(json.dumps(error_entry) + "\n")
             continue

        actual_generations = generations_list[0] # Get the list of samples

        if generation_index_to_select < 0 or generation_index_to_select >= len(actual_generations):
            error_count +=1
            logging.warning(f"[{instance_id}] Index {generation_index_to_select} out of bounds for {len(actual_generations)} available generations.")
            error_entry = { "model_name_or_path": args.model, "instance_id": instance_id, "test_patch": "", "raw_test_patch": f"INDEX_ERROR: Index {generation_index_to_select} not available (found {len(actual_generations)}).", "original_file_content": "", "error": "Index out of bounds" }
            with open(processed_output_file, "a", encoding='utf-8') as f: f.write(json.dumps(error_entry) + "\n")
            continue

        raw_generation_text = actual_generations[generation_index_to_select]
        if not raw_generation_text or "API_ERROR" in str(raw_generation_text): # Check if it's None or contains API_ERROR marker
            error_count += 1
            logging.warning(f"[{instance_id}] Generation at index {generation_index_to_select} is empty or contains API error.")
            empty_entry = { "model_name_or_path": args.model, "instance_id": instance_id, "test_patch": "", "raw_test_patch": str(raw_generation_text), "original_file_content": "", "error": "Empty or API Error generation" }
            with open(processed_output_file, "a", encoding='utf-8') as f: f.write(json.dumps(empty_entry) + "\n")
            continue

        # --- Extract Code ---
        logging.debug(f"[{instance_id}] Attempting extraction from raw_generation_text (index {generation_index_to_select}):\n{repr(raw_generation_text)[:200]}...")
        extracted_java_code = extract_first_java_code_block(raw_generation_text)

        git_diff_patch = ""
        error_msg = None
        if extracted_java_code:
            logging.debug(f"[{instance_id}] Extraction successful.")
            # --- Determine Target Path ---
            target_module = None
            # original_path is already fetched from original_code_patch_info above
            if original_path:
                # Use posixpath for consistent splitting even on Windows
                import posixpath
                path_parts = original_path.split(posixpath.sep)
                if len(path_parts) > 1:
                     target_module = path_parts[0]
                     logging.info(f"[{instance_id}] Inferred target module from path '{original_path}': {target_module}")
                else:
                     # Handle case where path is just a filename like "pom.xml" - no module subdir
                     logging.info(f"[{instance_id}] Path '{original_path}' has no directory components. Assuming no specific module subdir.")

            package_path = extract_package_from_java(extracted_java_code)
            test_file_name = "ReproduceBugTest.java" # As specified in prompt

            # Construct path using os.path.join for local system compatibility
            if not target_module:
                 logging.warning(f"[{instance_id}] Could not determine target module (or path had no dirs). Placing test in default 'repro_test'.")
                 test_dir_path_base = "repro_test" # Default directory
                 if package_path:
                     full_relative_path = os.path.join(test_dir_path_base, package_path.replace('.', os.path.sep), test_file_name)
                 else:
                     full_relative_path = os.path.join(test_dir_path_base, test_file_name)
            else:
                 test_src_root = os.path.join(target_module, "src", "test", "java")
                 if package_path:
                     full_relative_path = os.path.join(test_src_root, package_path.replace('.', os.path.sep), test_file_name)
                 else:
                     full_relative_path = os.path.join(test_src_root, test_file_name)

            # Convert final path to use forward slashes for Git patch consistency
            full_relative_path_git = full_relative_path.replace(os.path.sep, '/')
            logging.info(f"[{instance_id}] Calculated test file path for patch: {full_relative_path_git}")
            # --- End Determine Target Path ---

            git_diff_patch = create_patch_from_java_code(extracted_java_code, full_relative_path_git)
            processed_count += 1
        else:
            logging.debug(f"[{instance_id}] Extraction FAILED for index {generation_index_to_select}.")
            error_count += 1
            error_msg = f"Code extraction failed for generation index {generation_index_to_select}"

        # Write the processed data
        processed_entry = {
            "model_name_or_path": args.model, "instance_id": instance_id,
            "test_patch": git_diff_patch.lstrip(), # Remove potential leading whitespace
            "raw_test_patch": raw_generation_text, # Keep the raw text for reference
            "original_file_content": "", # Keep field for schema consistency, but it's empty here
        }
        if error_msg: processed_entry["error"] = error_msg
        with open(processed_output_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(processed_entry) + "\n")

    print(f"Post-processing complete for index {generation_index_to_select}. Processed: {processed_count}, Errors/Skipped: {error_count}, Missing Path Info: {missing_path_info_count}")
    
# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate Java reproduction tests using Gemini.")

    # Input/Output Arguments
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to store logs and output files.")
    parser.add_argument("--dataset_name", type=str, default="Daoguang/Multi-SWE-bench", help="Name of the Hugging Face dataset (still needed for problem statements).")
    parser.add_argument("--dataset_split", type=str, default="java_verified", help="Split of the dataset to use.")
    # --- ADDED ARGUMENT ---
    parser.add_argument("--filtered_patches_file", type=str, default="filtered_patches.jsonl", help="Path to the JSONL file containing filtered patches and instance_ids to process.")
    # --- END ADDED ARGUMENT ---

    # LLM Configuration
    parser.add_argument("--model", type=str, default="gemini-1.5-flash-latest", help="Gemini model name.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature for Gemini.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum output tokens for Gemini.")
    parser.add_argument("--max_samples", type=int, default=1, help="Number of test generations to attempt per issue.")

    # Processing Arguments
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for concurrent API calls.")
    parser.add_argument("--target_id", type=str, default=None, help="If set, only process this specific instance_id (must also be in filtered_patches_file).")
    parser.add_argument("--force_regenerate", action="store_true", help="Regenerate tests even if found in the output file.")
    parser.add_argument("--run_post_processing_only", action="store_true", help="If set, skip generation and only run post-processing.")

    args = parser.parse_args()

    # Basic logging config for the main process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define the raw output file path
    raw_output_filename = "output_raw_generations.jsonl"
    args.raw_output_file = os.path.join(args.output_folder, raw_output_filename)

    if not os.path.exists(args.filtered_patches_file):
         logging.error(f"Error: Filtered patches file not found at {args.filtered_patches_file}")
         print(f"Error: Filtered patches file not found at {args.filtered_patches_file}")
         return # Cannot proceed without the filtered list

    if args.run_post_processing_only:
        print("Running in POST-PROCESSING ONLY mode.")
        if not os.path.exists(args.raw_output_file):
            print(f"Error: Raw output file not found at {args.raw_output_file}.")
            return
        print("\nStarting post-processing phase...")
        # --- Pass args to post_process_tests ---
        for i in range(args.max_samples):
            print(f"\n--- Post-processing sample index {i} ---")
            args.select_id = i
            args.output_file = os.path.join(args.output_folder, f"output_{i}_processed_reproduction_test.jsonl")
            # Pass the full args object now
            post_process_tests(args)
        # --- End modification ---
        print("\nPost-processing phase complete.")
    else:
        print(f"Starting test generation using problem statements from '{args.dataset_name}' (split '{args.dataset_split}').")
        print(f"Processing instances listed in: {args.filtered_patches_file}")
        print(f"Output will be saved to: {args.output_folder}")
        args.output_file = args.raw_output_file # Set the output for generate_tests
        logs_dir = os.path.join(args.output_folder, "generating_test_logs")
        os.makedirs(logs_dir, exist_ok=True)
        # --- Pass args to generate_tests ---
        generate_tests(args) # Pass the full args object
        # --- End modification ---
        print("Test generation phase complete.")

        if os.path.exists(args.raw_output_file):
             print("\nStarting post-processing phase...")
             # --- Pass args to post_process_tests ---
             for i in range(args.max_samples):
                 print(f"\n--- Post-processing sample index {i} ---")
                 args.select_id = i
                 args.output_file = os.path.join(args.output_folder, f"output_{i}_processed_reproduction_test.jsonl")
                 # Pass the full args object now
                 post_process_tests(args)
             # --- End modification ---
             print("\nPost-processing phase complete.")
        else:
             print("\nSkipping post-processing phase as raw output file was not created (check generation logs/errors).")


    print(f"\nWorkflow complete. Check results in: {args.output_folder}")
    if not args.run_post_processing_only:
         print("Next step: Run run_reproduction_tests.py.") # Simplified next step
    else:
         print("Next step: Check the generated output_N_processed files, then run run_reproduction_tests.py.") # Simplified next step

if __name__ == "__main__":
    main()