import argparse
import json
import os
import subprocess
import re
import tempfile
import shutil
import time
import uuid
import traceback
from typing import List, Dict, Set, Tuple, Optional
from dotenv import load_dotenv

# Attempt to import necessary libraries
try:
    import jsonlines
except ImportError:
    print("Error: 'jsonlines' library not found. Please install it using: pip install jsonlines")
    exit(1)
try:
    # Check if docker library is installed and Docker daemon is running
    import docker
    try:
        docker_client = docker.from_env()
        docker_client.ping() # Check connection to Docker daemon
        print("Docker client initialized and connected.")
    except docker.errors.DockerException as e:
        print(f"Error: Docker library installed, but failed to connect to Docker daemon.")
        print(f"Ensure Docker is running. Details: {e}")
        exit(1)
except ImportError:
     print("Error: 'docker' library not found. Please install it using: pip install docker")
     exit(1)
try:
    import google.generativeai as genai
except ImportError:
    print("Error: 'google-generativeai' library not found. Please install it using: pip install google-generativeai")
    exit(1)


# --- LLM Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_ENABLED = bool(GEMINI_API_KEY) # Flag to track if LLM can be used

if not LLM_ENABLED:
    print("Warning: GEMINI_API_KEY environment variable not set. LLM suggestions will be disabled.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Safety settings to avoid blocking potentially relevant code/commands
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        print("Google Generative AI client configured.")
    except Exception as e:
        print(f"Error configuring Google Generative AI: {e}")
        print("LLM suggestions will be disabled.")
        LLM_ENABLED = False


# --- Test Output Parsing Functions ---

def parse_maven_surefire_stdout(log_data: str) -> List[str]:
    """Parses Maven Surefire stdout logs to find passing tests."""
    # (Implementation remains the same as previous version)
    passing_tests: Set[str] = set()
    running_tests: Set[str] = set()
    failed_tests: Set[str] = set()
    error_tests: Set[str] = set()
    running_test_class_regex = re.compile(r"\[INFO\] Running ([\w\.$]+)")
    # Updated regex to better capture failures/errors in different formats
    failure_regex = re.compile(r"\[ERROR\] Failures:\s*([\w\.$#]+)")
    error_regex = re.compile(r"\[ERROR\] Errors:\s*([\w\.$#]+)")
    # Regex for tests listed under T E S T S section failures/errors
    tests_section_fail_err_regex = re.compile(r"([\w\.$]+)\s+Time elapsed: [\d\.]+ s <<< (?:FAILURE|ERROR)!")
    # Regex for summary line
    summary_regex = re.compile(
        r"\[INFO\] Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)"
    )
    overall_success = False
    tests_run_count = 0

    for line in log_data.splitlines():
        match = running_test_class_regex.search(line)
        if match: running_tests.add(match.group(1))

        # Capture failures/errors from specific lines
        match = failure_regex.search(line)
        if match: failed_tests.add(match.group(1).split('#')[0]) # Add class name
        match = error_regex.search(line)
        if match: error_tests.add(match.group(1).split('#')[0]) # Add class name

        # Capture failures/errors from the "T E S T S" section summary
        match = tests_section_fail_err_regex.search(line)
        if match:
            # Extract the class name (might need refinement based on actual output)
            # Assuming format like com.package.ClassName.testMethod or just com.package.ClassName
            full_name = match.group(1)
            class_name = '.'.join(full_name.split('.')[:-1]) if '.' in full_name else full_name
            failed_tests.add(class_name) # Add class name

        # Check summary line for total counts
        match = summary_regex.search(line)
        if match:
            tests_run_count = int(match.group(1))
            # Consider overall success only if the final build summary says so

    if re.search(r"\[INFO\] BUILD SUCCESS", log_data):
        overall_success = True

    # Deduce passing tests: start with all run, remove known failed/errored
    passing_tests = running_tests - failed_tests - error_tests

    # Handle edge cases based on overall success and identified tests
    if overall_success and not passing_tests and running_tests:
        print("Warning: Maven BUILD SUCCESS reported, but no specific passing tests identified from 'Running...' lines minus failures/errors. Assuming all run tests passed (check logs carefully).")
        passing_tests = running_tests
    elif not overall_success and not failed_tests and not error_tests and running_tests:
        print("Warning: Maven BUILD FAILURE/ERROR reported, but no specific failed tests identified. Cannot determine passing tests reliably from stdout.")
        # Consider returning empty list if build failure means results are unreliable
        # passing_tests = set()

    print(f"Maven stdout parsing: Found {len(running_tests)} running classes, {len(failed_tests.union(error_tests))} failed/errored classes. Deduced {len(passing_tests)} passing classes.")
    return sorted(list(passing_tests))


def parse_gradle_stdout(log_data: str) -> List[str]:
    """Parses Gradle stdout logs to find passing tests."""
    # (Implementation remains the same as previous version)
    passing_tests: Set[str] = set()
    failed_tests: Set[str] = set()
    tests_executed = False
    build_successful = False
    # Regex to find lines indicating a test method passed/failed
    # Example: com.example.MyTestClass > myMethod() PASSED
    # Example: com.example.sub.MyOtherTest > anotherTest FAILED
    test_result_regex = re.compile(r"^([\w\.\$]+)\s+>\s+([\w\$]+)\(?\)?\s+(PASSED|FAILED)", re.MULTILINE)

    build_success_regex = re.compile(r"^BUILD SUCCESSFUL", re.MULTILINE)
    test_task_regex = re.compile(r"> Task :(\w+:)?test\b") # Matches :test, :subproject:test

    if test_task_regex.search(log_data): tests_executed = True
    if build_success_regex.search(log_data): build_successful = True

    for match in test_result_regex.finditer(log_data):
        test_id = f"{match.group(1)}.{match.group(2)}" # class.method
        result = match.group(3)
        if result == "PASSED":
            passing_tests.add(test_id)
        elif result == "FAILED":
            failed_tests.add(test_id)

    # Refine passing tests by removing any that also appear as failed (unlikely but safe)
    passing_tests = passing_tests - failed_tests

    if tests_executed and not build_successful:
        print("Warning: Gradle build failed. Passing tests identified before failure might be incomplete or misleading.")
    if build_successful and tests_executed and not passing_tests and not failed_tests:
        print("Warning: Gradle build successful and tests ran, but no specific PASSED/FAILED tests identified via regex. Test output format might differ or no tests were found.")

    print(f"Gradle stdout parsing: Found {len(passing_tests)} passed methods, {len(failed_tests)} failed methods.")
    return sorted(list(passing_tests))


# --- Build System & LLM Interaction ---

def get_llm_suggestions(repo_path: str, build_system: str) -> Optional[Tuple[str, str]]:
    """
    Uses Gemini to suggest the most appropriate test command and Docker image.

    Args:
        repo_path: Path to the local repository.
        build_system: Detected build system ('maven' or 'gradle').

    Returns:
        A tuple (suggested_test_command, suggested_docker_image) or None on failure.
    """
    if not LLM_ENABLED: # Check if LLM interaction is possible
        print("LLM suggestions disabled: API key not set or configuration failed.")
        return None

    print(f"Attempting to get LLM suggestions for {build_system} project...")
    # --- Construct Prompt ---
    prompt = f"Analyze the root directory structure and build file for a Java {build_system} project.\n\n"
    prompt += "Root Directory Structure (Top Level):\n"
    try:
        count = 0
        max_items = 30 # Limit number of files/dirs listed
        for item in sorted(os.listdir(repo_path)):
             item_path = os.path.join(repo_path, item)
             is_dir = os.path.isdir(item_path)
             # Ignore hidden files/dirs unless it's .mvn or .gradle
             if item.startswith('.') and item not in ['.mvn', '.gradle']:
                  continue
             prompt += f"- {item}{'/' if is_dir else ''}\n"
             count += 1
             if count >= max_items:
                  prompt += "- ... (more items exist)\n"
                  break
    except OSError as e:
        print(f"Warning: Could not list directory {repo_path}: {e}")
        prompt += "- (Could not list directory contents)\n"

    build_file_path = ""
    build_file_content = ""
    if build_system == "maven":
        build_file_path = os.path.join(repo_path, "pom.xml")
    elif build_system == "gradle":
        # Prefer build.gradle.kts if both exist? Or check wrapper properties?
        # For simplicity, check for build.gradle first.
        build_file_path = os.path.join(repo_path, "build.gradle")
        if not os.path.exists(build_file_path):
            build_file_path = os.path.join(repo_path, "build.gradle.kts")

    if build_file_path and os.path.exists(build_file_path):
        prompt += f"\nBuild File ({os.path.basename(build_file_path)} - First 100 lines):\n```\n"
        try:
            with open(build_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                build_file_content = "".join(f.readline() for _ in range(100))
                prompt += build_file_content
                if len(build_file_content.splitlines()) >= 100:
                     prompt += "\n... (truncated)"
        except Exception as e:
            print(f"Warning: Could not read build file {build_file_path}: {e}")
            prompt += "(Could not read build file content)"
        prompt += "\n```\n"
    else:
         prompt += "\nBuild File: Not found or not readable.\n"

    # Add Java version detection if possible (e.g., from pom.xml, build.gradle, .java-version)
    java_version_hint = " (unknown Java version)"
    # Basic check in common places
    try:
        if build_system == "maven" and build_file_content:
            match = re.search(r'<maven\.compiler\.(?:source|target)>(\d+\.?\d*)', build_file_content)
            if match: java_version_hint = f" (likely Java {match.group(1)})"
        elif build_system == "gradle" and build_file_content:
             match = re.search(r'sourceCompatibility\s*=\s*[\'"]? JavaVersion\.VERSION_(\d+)', build_file_content)
             if not match:
                  match = re.search(r'sourceCompatibility\s*=\s*[\'"]?(\d+\.?\d*)', build_file_content)
             if match: java_version_hint = f" (likely Java {match.group(1)})"
        # Could add check for .java-version file etc.
    except Exception:
        pass # Ignore errors during version hinting

    prompt += f"""
Based on this {build_system} project structure{java_version_hint}:
1. Suggest the most appropriate single command-line command to execute ALL tests using {build_system}. The command should run from the project root. It MUST include necessary flags/options to ensure the build continues even if some tests fail (e.g., for Maven: '-B -DskipTests=false -DfailIfNoTests=false -Dmaven.test.failure.ignore=true'; for Gradle: '--continue'). Consider if any specific profiles or modules seem necessary based on the structure (though standard execution is preferred if unsure).
2. Suggest a single, suitable, publicly available Docker image tag (e.g., 'maven:3.9-eclipse-temurin-17', 'gradle:8.5-jdk11') that contains the necessary JDK{java_version_hint} and {build_system} version for this project. Prioritize official images or widely used ones like eclipse-temurin.

Provide the output ONLY in the following JSON format (no explanations before or after):
{{
  "test_command": "suggested_command_here",
  "docker_image": "suggested_docker_image_tag_here"
}}
"""
    # --- Call LLM API ---
    try:
        model = genai.GenerativeModel('gemini-2.0-flash') # Using flash for speed/cost
        response = model.generate_content(prompt, safety_settings=safety_settings)
        response_text = response.text.strip()
        print(f"LLM Raw Response:\n{response_text}")

        # Clean potential markdown code block fences
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-len("```")].strip()

        # Attempt to parse the JSON
        data = json.loads(response_text)
        command = data.get("test_command")
        image = data.get("docker_image")

        if command and image:
            # Basic validation: command should contain the build tool name
            if (build_system == "maven" and "mvn" not in command) or \
               (build_system == "gradle" and ("gradle" not in command)):
                print(f"Warning: LLM suggested command '{command}' might be invalid for {build_system}. Check carefully.")
            # Basic validation: image should contain ':'
            if ":" not in image:
                 print(f"Warning: LLM suggested image '{image}' might be invalid format. Check carefully.")

            print(f"LLM Suggested Command: {command}")
            print(f"LLM Suggested Image: {image}")
            return command, image
        else:
            print("Warning: LLM response did not contain expected 'test_command' or 'docker_image' keys in JSON.")
            return None

    except json.JSONDecodeError:
        print(f"Warning: LLM response was not valid JSON: {response.text}")
        return None
    except Exception as e:
        print(f"Error during LLM suggestion generation: {e}")
        if hasattr(response, 'prompt_feedback'):
             print(f"Prompt Feedback: {response.prompt_feedback}")
        return None


def determine_test_config(repo_path: str) -> Tuple[str, str, str]:
    """
    Determines the build system, test command, and Docker image.
    Uses LLM for suggestions with file-based detection as fallback.
    """
    # (Implementation remains the same as previous version, calls updated get_llm_suggestions)
    build_system = None
    default_test_command = None
    default_docker_image = None

    # 1. Detect Build System and set initial defaults
    pom_path = os.path.join(repo_path, "pom.xml")
    gradle_path = os.path.join(repo_path, "build.gradle")
    gradle_kts_path = os.path.join(repo_path, "build.gradle.kts")

    if os.path.isfile(pom_path):
        build_system = "maven"
        default_test_command = "mvn test -B -DskipTests=false -DfailIfNoTests=false -Dmaven.test.failure.ignore=true"
        default_docker_image = "maven:3.9-eclipse-temurin-17"
        print("Detected Maven (pom.xml). Default command/image set.")
    elif os.path.isfile(gradle_path) or os.path.isfile(gradle_kts_path):
        build_system = "gradle"
        gradlew_path = os.path.join(repo_path, "gradlew")
        # Ensure gradlew is executable if it exists
        if os.path.isfile(gradlew_path):
             try:
                  os.chmod(gradlew_path, os.stat(gradlew_path).st_mode | 0o111) # Add execute permission
             except OSError as e:
                  print(f"Warning: Could not make gradlew executable: {e}")
        command_prefix = "./gradlew" if os.path.isfile(gradlew_path) else "gradle"
        default_test_command = f"{command_prefix} test --continue"
        default_docker_image = "gradle:8.5-jdk17"
        print("Detected Gradle (build.gradle/kts). Default command/image set.")
    else:
        raise ValueError("Could not detect Maven or Gradle build file in repository root.")

    # 2. Try getting LLM suggestions
    llm_suggestion = get_llm_suggestions(repo_path, build_system)

    # 3. Use LLM suggestions or fall back to defaults
    if llm_suggestion:
        suggested_command, suggested_image = llm_suggestion
        # Use suggested command unless it clearly doesn't match build system
        if build_system == "maven" and not suggested_command.startswith("mvn"):
             print("Warning: LLM suggested command doesn't start with 'mvn'. Using default.")
             final_command = default_test_command
        elif build_system == "gradle" and not (suggested_command.startswith("gradle") or suggested_command.startswith("./gradlew")):
             print("Warning: LLM suggested command doesn't start with 'gradle' or './gradlew'. Using default.")
             final_command = default_test_command
        else:
             final_command = suggested_command
        # Use suggested image
        final_image = suggested_image
    else:
        print("Using default test command and Docker image due to LLM suggestion failure or disabling.")
        final_command = default_test_command
        final_image = default_docker_image

    return build_system, final_command, final_image


# --- Docker Execution ---

def run_tests_in_docker(
    local_repo_path: str,
    commit_hash: str,
    test_command: str,
    instance_id: str,
    docker_image: str,
    timeout: int = 1800,
    run_id: str = "find_passing_java_tests_llm"
) -> List[str]:
    """Uses Docker to run tests from a locally mounted repository."""
    # (Implementation remains the same as previous version)
    passing_tests = []
    start_time = time.time()
    container_repo_path = "/usr/src/repo"

    log_dir = os.path.join("logs", run_id, instance_id)
    os.makedirs(log_dir, exist_ok=True)

    absolute_repo_path = os.path.abspath(local_repo_path)
    if not os.path.isdir(absolute_repo_path):
        print(f"Error: Local repository path does not exist or is not a directory: {absolute_repo_path}")
        return []

    docker_run_command = [
        "docker", "run", "--rm",
        "-v", f"{absolute_repo_path}:{container_repo_path}",
        "--workdir", container_repo_path,
        docker_image,
        "/bin/bash", "-c",
        f"echo '--- Verifying commit inside container ---' && git rev-parse HEAD && echo '--- Running test command ---' && {test_command}"
    ]

    print(f"\n--- Running Docker for instance: {instance_id} (Commit: {commit_hash}) ---")
    print(f"Host Repo Path: {absolute_repo_path}")
    print(f"Docker Image: {docker_image}")
    print(f"Test Command: {test_command}")

    try:
        process = subprocess.run(
            docker_run_command, capture_output=True, text=True, timeout=timeout,
            check=False, encoding='utf-8', errors='replace'
        )
        success = process.returncode == 0
        log_data = f"--- STDOUT ---\n{process.stdout}\n\n--- STDERR ---\n{process.stderr}"

        log_file_path = os.path.join(log_dir, "test_execution.log")
        with open(log_file_path, "w", encoding='utf-8') as f:
            f.write(f"Instance ID: {instance_id}\nCommit Hash: {commit_hash}\n")
            f.write(f"Docker Image: {docker_image}\nTest Command: {test_command}\n")
            f.write(f"Return Code: {process.returncode}\n")
            f.write(log_data)
        print(f"Docker execution log saved to: {log_file_path}")

        internal_commit_match = re.search(r'--- Verifying commit inside container ---\n([a-f0-9]+)', process.stdout)
        if internal_commit_match:
             internal_commit = internal_commit_match.group(1)
             if not commit_hash.startswith(internal_commit):
                  print(f"Warning: Commit hash inside container ({internal_commit[:8]}...) might not match expected ({commit_hash[:8]}...). Check mount/checkout.")
        else:
             print("Warning: Could not verify commit hash inside container from stdout.")

        if not success:
            print(f"Error: Docker container execution failed for {instance_id} with return code {process.returncode}.")
            return []

        print(f"Docker execution successful for {instance_id}. Parsing test results...")
        if "mvn" in test_command.lower():
            passing_tests = parse_maven_surefire_stdout(log_data)
        elif "gradle" in test_command.lower():
            passing_tests = parse_gradle_stdout(log_data)
        else:
            print(f"Warning: Unknown test command '{test_command}'. Cannot automatically parse results.")
            passing_tests = []

    except subprocess.TimeoutExpired:
        print(f"Error: Docker execution timed out for {instance_id} after {timeout} seconds.")
        log_file_path = os.path.join(log_dir, "test_execution.log")
        with open(log_file_path, "w", encoding='utf-8') as f: f.write("Result: Timeout\n")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred during Docker execution for {instance_id}: {e}")
        print(traceback.format_exc())
        log_file_path = os.path.join(log_dir, "test_execution.log")
        with open(log_file_path, "w", encoding='utf-8') as f: f.write(f"Result: Unexpected Error\n{traceback.format_exc()}\n")
        return []
    finally:
        end_time = time.time()
        print(f"--- Docker finished for instance: {instance_id} (Duration: {end_time - start_time:.2f}s) ---")

    return passing_tests


def main():
    parser = argparse.ArgumentParser(
        description="Find passing tests in a specific commit of a Java repository using Docker and LLM suggestions for config."
    )
    parser.add_argument("--repo_url", required=True, help="URL of the Git repository to clone.")
    parser.add_argument("--commit_hash", required=True, help="Commit hash to checkout and test.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--instance_id", required=True, help="Given Instance ID")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout for Docker execution.")
    parser.add_argument("--run_id", default=f"find_passing_java_tests_llm_{int(time.time())}", help="Unique ID for the run.")

    args = parser.parse_args()

    # --- Pre-checks ---
    if not shutil.which("docker"): print("Error: Docker command not found."); return
    if not shutil.which("git"): print("Error: git command not found."); return
    # Removed mandatory API key check here, will proceed with defaults if not set

    print("--- Starting LLM-Assisted Java Test Finder ---")
    # (Print args)

    temp_dir = None
    try:
        # --- Clone Repo Temporarily ---
        temp_dir = tempfile.mkdtemp(prefix="java_repo_test_")
        print(f"Cloning {args.repo_url} to temporary directory {temp_dir}...")
        # (Clone logic remains the same)
        clone_command = ["git", "clone", "--quiet", args.repo_url, temp_dir] # Added --quiet
        clone_process = subprocess.run(clone_command, capture_output=True, text=True, check=False)
        if clone_process.returncode != 0:
            print(f"Error cloning repository: {clone_process.stderr}")
            return
        print("Clone successful.")


        # --- Checkout Commit ---
        print(f"Checking out commit {args.commit_hash}...")
        # (Checkout logic remains the same)
        checkout_command = ["git", "checkout", "--force", args.commit_hash]
        checkout_process = subprocess.run(checkout_command, cwd=temp_dir, capture_output=True, text=True, check=False)
        if checkout_process.returncode != 0:
            print(f"Error checking out commit {args.commit_hash}: {checkout_process.stderr}")
            # Attempt to fetch if checkout fails (might be shallow clone or missing commit)
            print("Attempting git fetch...")
            fetch_command = ["git", "fetch", "--all", "--quiet"]
            subprocess.run(fetch_command, cwd=temp_dir, check=False)
            checkout_process = subprocess.run(checkout_command, cwd=temp_dir, capture_output=True, text=True, check=False)
            if checkout_process.returncode != 0:
                 print(f"Error checking out commit {args.commit_hash} even after fetch: {checkout_process.stderr}")
                 return
        print("Checkout successful.")

        # --- Determine Test Config (using temp repo path) ---
        build_system, test_command, docker_image = determine_test_config(temp_dir)
        print(f"Determined Build System: {build_system}")
        print(f"Using Test Command: {test_command}")
        print(f"Using Docker Image: {docker_image}")

        # --- Generate Instance ID ---
        # (Instance ID generation remains the same)
        repo_name = args.repo_url.split('/')[-1].replace('.git', '')
        instance_id =args.instance_id
        print(f"Instance ID: {instance_id}")


        # --- Execute Test Run (using temp repo path) ---
        # (Test execution logic remains the same)
        passing_tests = run_tests_in_docker(
            local_repo_path=temp_dir,
            commit_hash=args.commit_hash,
            test_command=test_command,
            instance_id=instance_id,
            docker_image=docker_image,
            timeout=args.timeout,
            run_id=args.run_id
        )


        # --- Save Results ---
        # (Saving results logic remains the same)
        output_data = {
            "instance_id": instance_id,
            "repo_url": args.repo_url,
            "commit_hash": args.commit_hash,
            "detected_build_system": build_system,
            "used_test_command": test_command,
            "used_docker_image": docker_image,
            "tests_passing_in_original_repo": passing_tests,
        }
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with jsonlines.open(args.output_file, mode="a") as writer:
            writer.write(output_data)
        print(f"\nResults for {instance_id} appended to {args.output_file}")
        if not passing_tests:
             print(f"Warning: No passing tests were identified for {instance_id}. Check test_execution.log in logs/{args.run_id}/{instance_id}/")


    except ValueError as e:
         print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main execution: {e}")
        print(traceback.format_exc())
    finally:
        # --- Cleanup Temporary Directory ---
        # (Cleanup logic remains the same)
        if temp_dir and os.path.exists(temp_dir):
            try:
                # Workaround for potential permission issues on Windows
                def remove_readonly(func, path, excinfo):
                    os.chmod(path, 0o777) # Make writable
                    func(path)
                shutil.rmtree(temp_dir, onerror=remove_readonly if os.name == 'nt' else None)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except OSError as e:
                print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")


    print("--- LLM-Assisted Java Test Finder Finished ---")


if __name__ == "__main__":
    main()
