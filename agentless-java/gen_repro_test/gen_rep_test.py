# filename: run_java_repro_generation_and_verification.py

import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from threading import Lock # Lock is kept for file writing
from typing import List, Dict, Any, Tuple, Set

# Attempt to import required libraries
try:
    from datasets import load_dataset
    from tqdm import tqdm
    import tiktoken
except ImportError:
    print("Error: Missing required libraries.")
    print("Please install them using: pip install datasets tqdm tiktoken")
    exit(1)

# Attempt to import google.generativeai
try:
    import google.generativeai as genai
    # Check for API key early (using the correct name)
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the environment variable with your API key.")
        exit(1)
    # Configure with the correct name
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except ImportError:
    print("Error: google-generativeai library not found.")
    print("Please install it using: pip install google-generativeai")
    exit(1)
except Exception as e:
    print(f"Error configuring Google Generative AI: {e}")
    exit(1)


# --- Utility Functions ---

def num_tokens_from_messages(message: str, model: str = "gemini-1.5-flash-latest") -> int:
    """Estimates token count for Gemini (approximation)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(message))
    except Exception:
        return len(message) // 4

# Setup logger (Modified to accept logger instance)
def setup_logger(log_file: str) -> logging.Logger:
    """Sets up a logger that writes to a file."""
    logger = logging.getLogger(log_file)
    for hdlr in logger.handlers[:]: logger.removeHandler(hdlr); hdlr.close()
    logger.setLevel(logging.INFO)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def load_jsonl(filepath: str) -> List[Dict]:
    """Loads a JSONL file into a list of dictionaries."""
    if not os.path.exists(filepath): return []
    data = []
    try:
        with open(filepath, "r", encoding='utf-8') as file:
            for i, line in enumerate(file):
                try: data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e: print(f"Warning: Skipping invalid JSON line {i+1} in {filepath}: {e}")
    except IOError as e: print(f"Error reading file {filepath}: {e}")
    return data

# Keep run_command utility
def run_command(command: List[str], cwd: str | None = None, logger: logging.Logger = None, check: bool = True, timeout: int | None = None, input_data: str | None = None) -> subprocess.CompletedProcess:
    """Runs a generic command, logs, captures output, handles input."""
    log_prefix = f"[{os.path.basename(cwd)}]" if cwd else ""
    cmd_str = ' '.join(command)
    if logger: logger.info(f"{log_prefix} Running command: {cmd_str}{f' in {cwd}' if cwd else ''}")
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, encoding='utf-8',
            check=check, timeout=timeout, input=input_data
        )
        if logger:
            stdout = result.stdout.strip() if result.stdout else ""
            stderr = result.stderr.strip() if result.stderr else ""
            if check and result.returncode != 0:
                 logger.error(f"{log_prefix} Command failed code {result.returncode}: {cmd_str}")
                 if stdout: logger.error(f"{log_prefix} stdout:\n{stdout}")
                 if stderr: logger.error(f"{log_prefix} stderr:\n{stderr}")
            else:
                 if stdout: logger.debug(f"{log_prefix} Command stdout:\n{stdout}")
                 if stderr: logger.debug(f"{log_prefix} Command stderr:\n{stderr}")
        return result
    except FileNotFoundError as e:
        if logger: logger.error(f"{log_prefix} Error: Command not found - {command[0]}. Is it installed and in PATH?")
        raise e
    except subprocess.CalledProcessError as e:
         if logger:
              logger.error(f"{log_prefix} Command failed: {e}")
              stdout = e.stdout.strip() if e.stdout else ""
              stderr = e.stderr.strip() if e.stderr else ""
              if stdout: logger.error(f"{log_prefix} stdout:\n{stdout}")
              if stderr: logger.error(f"{log_prefix} stderr:\n{stderr}")
         raise e
    except subprocess.TimeoutExpired as e:
         if logger: logger.error(f"{log_prefix} Command timed out after {timeout}s: {cmd_str}")
         raise e
    except Exception as e:
        if logger: logger.error(f"{log_prefix} An unexpected error occurred running command: {e}", exc_info=True)
        raise e


# --- LLM Interaction Class --- (Unchanged)

class GeminiChatDecoder:
    """Simple wrapper for Gemini API chat completions."""
    def __init__(
        self,
        model_name: str,
        logger: logging.Logger,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: int = 5, # seconds
    ):
        self.model_name = model_name; self.logger = logger; self.max_new_tokens = max_new_tokens
        self.temperature = temperature; self.retry_attempts = retry_attempts; self.retry_delay = retry_delay
        try: self.model = genai.GenerativeModel(model_name)
        except Exception as e: self.logger.error(f"Failed to initialize Gemini model '{model_name}': {e}"); raise

    def _request_gemini(self, message: str) -> Dict[str, Any]:
        """Makes a single request to the Gemini API with retries."""
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                if not hasattr(self, 'model'): raise RuntimeError("Gemini model not initialized.")
                chat = self.model.start_chat(history=[])
                response = chat.send_message(content=message, generation_config={"temperature": self.temperature, "max_output_tokens": self.max_new_tokens})
                prompt_tokens = num_tokens_from_messages(message, self.model_name)
                completion_tokens = num_tokens_from_messages(response.text, self.model_name)
                return {"response": response.text, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}, "finish_reason": "stop", "raw_response": response}
            except Exception as e:
                last_exception = e; self.logger.warning(f"Gemini API request failed (Attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1: time.sleep(self.retry_delay)
                else: self.logger.error(f"Gemini API request failed after {self.retry_attempts} attempts."); return {"response": "", "usage": {}, "error": str(e)}
        return {"response": "", "usage": {}, "error": str(last_exception)} # Fallback

    def codegen(self, message: str, num_samples: int = 1) -> List[Dict[str, Any]]:
        """Generates code samples using the Gemini model."""
        if self.temperature == 0 and num_samples > 1: self.logger.warning("Temp is 0, generating 1 sample."); num_samples = 1
        elif self.temperature > 0 and num_samples > 1: self.logger.info(f"Generating {num_samples} samples with temp={self.temperature}...")
        else: self.logger.info(f"Generating {num_samples} sample(s)...")
        trajs = []
        for i in range(num_samples):
             self.logger.debug(f"Requesting sample {i+1}/{num_samples}")
             result = self._request_gemini(message)
             if "error" in result: self.logger.error(f"Failed sample {i+1} due to: {result['error']}"); trajs.append({"response": "", "usage": result.get("usage", {}), "error": result["error"]})
             else: trajs.append(result)
        return trajs


# --- Test Generation Logic --- (Unchanged prompt, extraction, patch creation)

generate_tests_prompt_template = """
You are an expert Java test generation assistant.
Your task is to generate a JUnit 5 test class to reproduce a specific software bug described in a GitHub issue.

**GitHub Issue Description:**
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

**Instructions:**

1.  **Analyze the Issue:** Understand the bug described in the issue statement. Identify the conditions required to trigger it.
2.  **Generate JUnit 5 Test Class:** Create a *complete*, *compilable* Java class using JUnit 5 (`org.junit.jupiter.api.*`).
    * Include necessary `import` statements.
    * Define a `public` class (e.g., `ReproduceBugTest`). Do **not** include a `package` declaration.
    * Create at least one `public` test method annotated with `@Test`.
    * The test method(s) should contain the precise logic to trigger the bug described in the issue.
3.  **Output Requirements:** The test method(s) MUST print **exactly one** of the following messages to standard output based on the execution outcome:
    * `"Issue reproduced"`: If the test successfully triggers the described bug (e.g., catches the expected exception, observes the incorrect behavior).
    * `"Issue resolved"`: If the test runs *without* triggering the bug, indicating the bug might be fixed in the environment where the test is run (e.g., the expected exception is *not* thrown, the correct behavior is observed).
    * `"Other issues"`: If the test encounters an unexpected error (e.g., different exception, setup failure, ambiguous outcome). Print any unexpected exception stack traces to standard *error*.
4.  **Completeness:** Ensure the generated code is self-contained and ready to be compiled and executed within the context of the target project's classpath.
5.  **Format:** Wrap the *entire* Java test class in a single ```java ... ``` markdown block.

**Example Structure:**

```java
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
// Potentially import classes from the project being tested

public class ReproduceSpecificBugTest {{ // Choose a descriptive name

    @Test
    void testBuggyScenario() {{
        try {{
            // 1. Setup code to reach the buggy state (if necessary)
            // 2. Code that directly triggers the bug based on the issue description
            Object result = callCodeThatShouldFail();

            // 3. If the bug involves WRONG behavior rather than an exception, assert that here.
            //    If the wrong behavior occurs, print "Issue reproduced".
            //    If the correct behavior occurs, print "Issue resolved".
             System.out.println("Issue resolved"); // Default if no exception/wrong behavior detected

        }} catch (ExpectedBugException e) {{
            // 4. If the bug involves a SPECIFIC exception being thrown
            System.out.println("Issue reproduced");
        }} catch (Exception unexpectedException) {{
            // 5. Catch any OTHER exceptions during the test
            System.err.println("Test failed with unexpected exception:");
            unexpectedException.printStackTrace(System.err); // Print stack trace to stderr
            System.out.println("Other issues");
        }}
    }}

    // Helper methods (if needed)
    private Object callCodeThatShouldFail() throws Exception {{
        // Replace with actual method calls from the target project
        throw new RuntimeException("Placeholder: Replace with actual call"); // Placeholder
    }}
}}
```
Now, generate the Java test class for the provided issue.
"""

def extract_first_code_block(text: str) -> str | None:
    m = re.search(r"```java\s*(.*?)\s*```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if ("class " in code or "interface " in code or "enum " in code) and \
           ("public " in code or "@Test" in code) and \
           "import " in code and code.endswith("}"):
            print("Warning: Found code block without 'java' tag, assuming Java based on content.")
            return code
    print("Warning: Could not find a ```java ... ``` code block.")
    return None

def create_patch_from_code(java_code: str, logger: logging.Logger) -> Tuple[str | None, str | None]:
    if not java_code: return None, None
    m_cls = re.search(r"public\s+(?:final\s+|abstract\s+)?class\s+(\w+)", java_code)
    class_name = m_cls.group(1) if m_cls else "ReproduceBug"
    if not m_cls: logger.warning(f"Could not find public class declaration. Using default name: {class_name}.")
    m_pkg = re.search(r"^\s*package\s+([\w\.]+)\s*;", java_code, re.MULTILINE)
    pkg_path = m_pkg.group(1).replace(".", "/") + "/" if m_pkg else ""
    if m_pkg: logger.info(f"Detected package: {m_pkg.group(1)}")
    else: logger.info("No package declaration detected.")
    relative_file_path = f"src/test/java/{pkg_path}{class_name}.java"
    logger.info(f"Target test file path: {relative_file_path}")
    patch_header = (f"diff --git a/{relative_file_path} b/{relative_file_path}\n"
                    f"new file mode 100644\nindex 0000000..e69de29\n")
    patch_body = ["--- /dev/null", f"+++ b/{relative_file_path}"]
    java_code_cleaned = java_code.rstrip(); code_lines = java_code_cleaned.split('\n')
    if not code_lines or (len(code_lines) == 1 and not code_lines[0]):
        logger.warning("Generated Java code is empty after stripping whitespace."); patch_body.append("@@ -0,0 +0,0 @@")
    else:
        patch_body.append(f"@@ -0,0 +1,{len(code_lines)} @@")
        for line in code_lines: patch_body.append(f"+{line}")
    patch = patch_header + "\n".join(patch_body) + "\n"
    return patch, relative_file_path

def generate_instance_test(instance: Dict, args: argparse.Namespace, logger: logging.Logger) -> Dict:
    """Generates test code for a single instance."""
    instance_id = instance["instance_id"]
    logger.info(f"Starting test generation for {instance_id}")
    problem_statement = instance.get("problem_statement", "") or f"Issue Title: {instance.get('title', '')}\n\nIssue Body:\n{instance.get('body', '')}".strip()
    if not problem_statement: logger.error(f"No problem statement for {instance_id}."); return {"instance_id": instance_id, "status": "generation_failed", "error": "No problem statement found.", "patch": None}
    prompt = generate_tests_prompt_template.format(problem_statement=problem_statement)
    logger.debug(f"Formatted prompt (first 500 chars): {prompt[:500]}...")
    try: gemini_decoder = GeminiChatDecoder(model_name=args.model, logger=logger, max_new_tokens=args.max_tokens, temperature=args.temperature if args.max_samples > 1 else 0.0)
    except Exception as e: logger.error(f"LLM init failed: {e}"); return {"instance_id": instance_id, "status": "generation_failed", "error": f"LLM init failed: {e}", "patch": None}
    num_to_generate = args.max_samples if args.temperature > 0 else 1
    generation_results = gemini_decoder.codegen(prompt, num_samples=num_to_generate)
    if not generation_results or all("error" in r for r in generation_results): logger.error(f"LLM generation failed for {instance_id}."); return {"instance_id": instance_id, "status": "generation_failed", "error": "LLM returned no valid responses.", "all_results": generation_results, "patch": None}

    best_patch, best_test_file_path, selected_generation_result = None, None, None
    for i, result in enumerate(generation_results):
         if "error" in result or not result.get("response"): logger.warning(f"Sample {i+1} failed/empty for {instance_id}."); continue
         logger.info(f"Processing sample {i+1} for {instance_id}."); raw_code_output = result["response"]
         extracted_code = extract_first_code_block(raw_code_output)
         if extracted_code:
             logger.info(f"Extracted code block from sample {i+1} for {instance_id}.")
             patch, test_file_path = create_patch_from_code(extracted_code, logger)
             if patch and test_file_path: logger.info(f"Created patch for sample {i+1} targeting {test_file_path}."); best_patch, best_test_file_path, selected_generation_result = patch, test_file_path, result; break
             else: logger.warning(f"Failed patch creation from sample {i+1}, instance {instance_id}.")
         else: logger.warning(f"Could not extract code block from sample {i+1} for {instance_id}.")

    if best_patch:
        logger.info(f"Selected patch from sample {generation_results.index(selected_generation_result)+1} for {instance_id}.")
        return {"instance_id": instance_id, "status": "generation_success", "generation_result": selected_generation_result, "all_results": generation_results, "patch": best_patch, "test_file_path": best_test_file_path}
    else:
        logger.error(f"Failed patch generation for {instance_id} after {len(generation_results)} attempts."); return {"instance_id": instance_id, "status": "generation_failed", "error": "No valid code/patch.", "all_results": generation_results, "patch": None}


# --- Repository Management ---

def clone_repo(repo_url, base_commit, clone_dir, logger):
    """
    Clones a repository at a specific commit.

    Args:
        repo_url: Repository URL in format 'org/repo'
        base_commit: Git commit hash to checkout
        clone_dir: Directory to clone into

    Returns:
        Path to the cloned repository
    """
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)

    repo_name = repo_url.split("/")[-1]  # Extract repo name
    repo_path = os.path.join(clone_dir, repo_name)

    if os.path.exists(repo_path):
        logger.info(f"Repository {repo_name} already exists, skipping clone...")
    else:
        logger.info(f"Cloning {repo_url} at commit {base_commit}...")
        try:
            result = subprocess.run(
                ["git", "clone", f"https://github.com/{repo_url}.git", repo_path],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning repository: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return None

    # Checkout the base commit to match dataset state
    try:
        # Clean untracked files before checkout to avoid conflicts
        subprocess.run(
            ["git", "-C", repo_path, "clean", "-fd"],
            check=True
        )

        # Reset any changes
        subprocess.run(
            ["git", "-C", repo_path, "reset", "--hard"],
            check=True
        )

        # Checkout the base commit
        subprocess.run(
            ["git", "-C", repo_path, "checkout", base_commit],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking out commit {base_commit}: {e}")
        return None

    return repo_path


# --- Local Verification Logic ---

def detect_build_tool(repo_path: str, logger: logging.Logger) -> str | None:
    """Detects if a repository uses Maven or Gradle."""
    if os.path.exists(os.path.join(repo_path, "pom.xml")): logger.info("Detected build tool: Maven"); return "maven"
    if os.path.exists(os.path.join(repo_path, "build.gradle")) or os.path.exists(os.path.join(repo_path, "build.gradle.kts")): logger.info("Detected build tool: Gradle"); return "gradle"
    logger.warning(f"Could not detect pom.xml or build.gradle[.kts] in {repo_path}"); return None

def apply_patch(repo_path: str, patch: str, logger: logging.Logger) -> bool:
    """Applies a patch string using git apply. Assumes repo is already at base commit."""
    if not patch: logger.warning("Attempted to apply an empty patch."); return False
    logger.info("Attempting to apply patch...")
    try:
        # Apply patch - repo should be clean from clone_repo checkout
        run_command(["git", "apply", "--verbose", "--whitespace=fix", "--allow-empty"], input_data=patch, cwd=repo_path, logger=logger, check=True)
        logger.info("Patch applied successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to apply patch. Return code: {e.returncode}")
        return False
    except Exception as e:
         logger.error(f"An unexpected error occurred during patch application: {e}")
         return False

def run_test_and_capture_output(repo_path: str, test_fqn: str, build_tool: str, logger: logging.Logger, timeout: int = 300) -> Tuple[bool, str, str]:
    """Runs a specific test using Maven or Gradle and captures output."""
    success = False; stdout_str = ""; stderr_str = ""; cmd = []
    if not re.match(r"^([\w\.]+\.)+\w+$", test_fqn): logger.error(f"Invalid test FQN: {test_fqn}"); return success, stdout_str, f"Invalid FQN: {test_fqn}"
    logger.info(f"Running test: {test_fqn} using {build_tool}")
    if build_tool == "maven":
        class_name = test_fqn.split('.')[-1]
        cmd = ["mvn", "-B", "clean", "test-compile", "test", f"-Dtest={test_fqn}", "-DfailIfNoTests=false", f"-Dmaven.repo.local={os.path.join(os.path.expanduser('~'), '.m2', 'repository')}"]
    elif build_tool == "gradle":
        gradlew_path = os.path.join(repo_path, "gradlew")
        gradle_exec = gradlew_path if os.path.exists(gradlew_path) else "gradle"
        if os.path.exists(gradlew_path): os.chmod(gradlew_path, 0o755)
        cmd = [gradle_exec, "test", "--tests", test_fqn, "--quiet", "--continue"]
    else: logger.error(f"Unsupported build tool: {build_tool}"); return success, stdout_str, f"Unsupported build tool: {build_tool}"

    logger.info(f"Executing test command: {' '.join(cmd)}")
    try:
        result = run_command(cmd, cwd=repo_path, logger=logger, check=False, timeout=timeout)
        stdout_str = result.stdout; stderr_str = result.stderr
        success = result.returncode == 0
        if not success: logger.warning(f"Build/Test command execution failed with return code {result.returncode}.")
    except subprocess.TimeoutExpired: logger.error(f"Test execution timed out after {timeout} seconds for {test_fqn}"); stderr_str = f"TimeoutExpired"; success = False
    except Exception as e: logger.error(f"An unexpected error during test execution for {test_fqn}: {e}"); stderr_str = f"Execution Error: {e}"; success = False
    return success, stdout_str, stderr_str

def verify_reproduction(
    instance_id: str,
    repo_path: str, # Assumed to be already at the correct base_commit
    base_commit: str, # Passed for logging/confirmation
    test_patch: str,
    test_file_path: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Applies patch, runs test, checks output for 'Issue reproduced'.
    Assumes repo_path is already checked out to the correct base_commit by clone_repo.
    """
    logger.info(f"Starting verification for {instance_id} (expecting commit {base_commit[:7]}) in {repo_path}")
    verification_status = "verification_failed"; details = ""; stdout = ""; stderr = ""; build_tool = None

    # Verify current commit (optional but recommended)
    try:
        result = run_command(["git", "rev-parse", "HEAD"], cwd=repo_path, logger=logger, check=True, timeout=10)
        current_commit = result.stdout.strip()
        if current_commit != base_commit:
            logger.error(f"Repo at wrong commit! Expected {base_commit}, found {current_commit}. Aborting verification.")
            return {"status": "verification_failed", "error": "Repo not at expected base commit"}
    except Exception as e_revparse:
         logger.error(f"Could not verify current commit: {e_revparse}")
         return {"status": "verification_failed", "error": "Failed to verify repo commit"}

    if not test_patch or not test_file_path: return {"status": verification_status, "error": "Patch or path missing"}

    try:
        if not apply_patch(repo_path, test_patch, logger): return {"status": verification_status, "error": "Failed to apply test patch"}
        build_tool = detect_build_tool(repo_path, logger)
        if not build_tool: return {"status": verification_status, "error": "Failed to detect build tool"}

        if not test_file_path.startswith("src/test/java/") or not test_file_path.endswith(".java"):
             logger.error(f"Invalid test file path format: {test_file_path}"); return {"status": verification_status, "error": "Invalid test path format"}
        fqn_parts = test_file_path[len("src/test/java/"): -len(".java")]; test_fqn = fqn_parts.replace('/', '.')
        logger.info(f"Determined test FQN: {test_fqn}")

        build_success, stdout, stderr = run_test_and_capture_output(repo_path, test_fqn, build_tool, logger)
        stdout_clean = stdout.strip().replace('\r\n', '\n')

        if '"Issue reproduced"' in stdout_clean:
             logger.info(f"Verification successful: 'Issue reproduced' found for {instance_id}."); verification_status = "reproduced"; details = "'Issue reproduced' found."
        elif '"Issue resolved"' in stdout_clean:
             logger.warning(f"Verification failed: Found 'Issue resolved' for {instance_id}."); verification_status = "resolved_instead"; details = "'Issue resolved' found."
        elif '"Other issues"' in stdout_clean:
             logger.warning(f"Verification failed: Found 'Other issues' for {instance_id}."); verification_status = "other_issues"; details = "'Other issues' found."
        else:
             logger.warning(f"Verification failed: Required output not found for {instance_id}."); verification_status = "verification_failed"; details = f"Required output not found. Build success: {build_success}."

    except Exception as e:
        logger.error(f"Verification failed for {instance_id} due to unexpected error: {e}", exc_info=True); details = f"Unexpected error: {e}"; verification_status = "verification_failed"

    # No cleanup needed here, clone_repo handles repo state before checkout for the next instance.
    return {"status": verification_status, "details": details, "stdout": stdout, "stderr": stderr, "build_tool": build_tool}


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate and locally verify Java reproduction tests for Multi-SWE-bench.")
    # All arguments are optional now
    parser.add_argument("--dataset", default="Daoguang/Multi-SWE-bench", help="Hugging Face dataset name (default: Daoguang/Multi-SWE-bench).")
    parser.add_argument("--split", default="java_verified", help="Dataset split to use (default: java_verified).")
    parser.add_argument("--output_folder", default="./output_java_repro_tests", help="Folder for logs and results (default: ./output_java_repro_tests).")
    parser.add_argument("--repos_base_path", default="./java_repos", help="Base path to clone/manage repositories (default: ./java_repos).")
    parser.add_argument("--model", default="gemini-1.5-flash-latest", help="Gemini model name (default: gemini-1.5-flash-latest).")
    parser.add_argument("--max_samples", type=int, default=1, help="Number of test variations to generate per instance (default: 1).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation (default: 0.0). Set > 0 for multiple samples.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens for LLM response (default: 2048).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (WARNING: High values with user's clone_repo may cause issues). Default: 4.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if found.")
    parser.add_argument("--target_id", type=str, default=None, help="Optional: Process only a specific instance ID.")
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_folder, exist_ok=True); os.makedirs(args.repos_base_path, exist_ok=True)
    main_log_file = os.path.join(args.output_folder, "main_run.log")
    logger = setup_logger(main_log_file)
    results_file = os.path.join(args.output_folder, "reproduction_results.jsonl")
    generation_details_file = os.path.join(args.output_folder, "generation_details.jsonl")
    logger.info("Starting script with arguments:"); [logger.info(f"  {arg}: {value}") for arg, value in vars(args).items()]
    if not os.environ.get("GEMINI_API_KEY"): logger.error("GEMINI_API_KEY not set."); print("Error: GEMINI_API_KEY not set."); exit(1)
    if args.num_workers > 1: logger.warning("Running with multiple workers. The provided clone_repo function has potential concurrency issues (race conditions on clone check, no fetch on existing repos).")

    # --- Load Data ---
    logger.info(f"Loading dataset {args.dataset}, split {args.split}...")
    try: swe_bench_data = list(load_dataset(args.dataset, split=args.split, trust_remote_code=True)); logger.info(f"Loaded {len(swe_bench_data)} instances.")
    except Exception as e: logger.error(f"Failed dataset load: {e}", exc_info=True); print(f"Error loading dataset: {e}"); exit(1)

    # --- Assert Instance Count ---
    if args.dataset == "Daoguang/Multi-SWE-bench" and args.split == "java_verified":
        expected_instances = 91; actual_instances = len(swe_bench_data)
        if actual_instances != expected_instances: logger.error(f"Assertion Failed: Expected {expected_instances}, found {actual_instances}.")
        else: logger.info(f"Assertion Passed: Found expected {expected_instances} instances.")

    # --- Load Previous Results ---
    previous_results = {}
    if not args.overwrite and os.path.exists(results_file):
        logger.info(f"Loading previous results from {results_file}")
        prev_data = load_jsonl(results_file)
        previous_results = {item["instance_id"]: item for item in prev_data if "instance_id" in item}
        logger.info(f"Loaded results for {len(previous_results)} instances.")

    # --- Filter Instances ---
    if args.target_id:
         instance = next((inst for inst in swe_bench_data if inst.get("instance_id") == args.target_id), None)
         if instance: instances_to_process = [instance]; logger.info(f"Targeting specific instance: {args.target_id}")
         else: logger.error(f"Target instance ID {args.target_id} not found."); print(f"Error: Target instance {args.target_id} not found."); exit(1)
    else: instances_to_process = [inst for inst in swe_bench_data if args.overwrite or inst.get("instance_id") not in previous_results]
    logger.info(f"Processing {len(instances_to_process)} instances ({len(previous_results)} previously processed).")

    if not instances_to_process: logger.info("No new instances."); print("No new instances."); print_summary(results_file); return

    # --- Clear old results if overwriting ---
    if args.overwrite:
        for f_path in [results_file, generation_details_file]:
            if os.path.exists(f_path):
                 logger.warning(f"Overwriting existing file: {f_path}")
                 try: os.remove(f_path)
                 except OSError as e: logger.error(f"Failed to remove file {f_path}: {e}")

    # --- Process Instances ---
    results_lock = Lock() # For writing results/details files safely

    def process_instance(instance):
        instance_id = instance.get("instance_id"); instance_logger = setup_logger(os.path.join(args.output_folder, "instance_logs", f"{instance_id}.log"))
        instance_result = {"instance_id": instance_id}; repo_path = None

        try:
            # Step 1: Setup repo using user's clone_repo function
            # Assuming 'repo' field in dataset contains 'org/repo' slug
            repo_url_slug = instance.get("repo")
            #print(repo_url_slug)
            base_commit = instance.get("base_commit")

            if not repo_url_slug or not base_commit:
                instance_logger.error("Missing repo slug or base_commit information.")
                instance_result.update({"status": "repo_setup_failed", "error": "Missing repo/commit info", "verification_status": "skipped_repo_setup_failed"})
            else:
                # Call the user's provided clone_repo function
                repo_path = clone_repo(repo_url_slug, base_commit, args.repos_base_path, instance_logger)
                if not repo_path:
                    instance_logger.error(f"Failed to setup repository {repo_url_slug} at commit {base_commit[:7]}.")
                    # Ensure status reflects failure from clone_repo
                    instance_result.update({"status": "repo_setup_failed", "error": "Repo setup failed via clone_repo", "verification_status": "skipped_repo_setup_failed"})
                else:
                     instance_result["repo_path"] = repo_path # Record path if setup succeeded

            # Step 2: Generate Test Patch (always attempt)
            gen_data = generate_instance_test(instance, args, instance_logger)
            instance_result.update(gen_data)
            # Write generation details
            try:
                 with results_lock:
                      with open(generation_details_file, "a", encoding='utf-8') as gdf:
                           serializable_gen_data = {k: v for k, v in gen_data.items() if k not in ['raw_response', 'generation_result']}
                           if gen_data.get("generation_result"): serializable_gen_data["generation_result_summary"] = {"response_length": len(gen_data["generation_result"].get("response","")), "usage": gen_data["generation_result"].get("usage"), "error": gen_data["generation_result"].get("error")}
                           gdf.write(json.dumps(serializable_gen_data) + "\n")
            except Exception as e: instance_logger.error(f"Failed to write generation details: {e}")

            # Step 3: Verify only if repo setup succeeded AND generation succeeded
            if repo_path and gen_data["status"] == "generation_success":
                verification_data = verify_reproduction(instance_id, repo_path, base_commit, gen_data["patch"], gen_data["test_file_path"], instance_logger)
                instance_result.update(verification_data)
            elif not repo_path: instance_result.setdefault("verification_status", "skipped_repo_setup_failed")
            else: instance_result.setdefault("verification_status", "skipped_generation_failed")

        except Exception as e:
             instance_logger.error(f"Unexpected error processing instance {instance_id}: {e}", exc_info=True)
             instance_result.update({"status": "processing_error", "error": str(e)})
             instance_result.setdefault("verification_status", "skipped_processing_error")

        # Write final result summary
        try:
            with results_lock:
                 with open(results_file, "a", encoding='utf-8') as rf:
                      summary_result = {k: instance_result.get(k) for k in ["instance_id", "generation_status", "verification_status", "error", "details", "build_tool"]}
                      rf.write(json.dumps(summary_result) + "\n")
        except Exception as e: instance_logger.error(f"Failed to write final result: {e}")
        return instance_result

    # Execute processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_instance, inst) for inst in instances_to_process]
        progress_bar = tqdm(total=len(instances_to_process), desc="Processing Instances", unit="instance")
        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception as e: logger.error(f"Error retrieving result from worker: {e}", exc_info=True)
            finally: progress_bar.update(1)
        progress_bar.close()

    # --- Final Evaluation Summary ---
    logger.info("Processing complete. Generating final summary.")
    print_summary(results_file)
    logger.info("--- Script Finished ---")

def print_summary(results_file_path):
    """Loads results and prints a formatted summary."""
    print("\n--- Evaluation Summary ---")
    if not os.path.exists(results_file_path): print("Results file not found."); return
    final_results_data = load_jsonl(results_file_path)
    if not final_results_data: print("No results found in results file."); return

    latest_results = {res["instance_id"]: res for res in final_results_data if "instance_id" in res}
    total_unique_attempted = len(latest_results)

    gen_success = sum(1 for res in latest_results.values() if res.get("generation_status") == "generation_success")
    repo_fail = sum(1 for res in latest_results.values() if res.get("status") == "repo_setup_failed" or res.get("verification_status") == "skipped_repo_setup_failed")
    proc_error = sum(1 for res in latest_results.values() if res.get("status") == "processing_error")

    reproduced = sum(1 for res in latest_results.values() if res.get("verification_status") == "reproduced")
    resolved_instead = sum(1 for res in latest_results.values() if res.get("verification_status") == "resolved_instead")
    other_issues = sum(1 for res in latest_results.values() if res.get("verification_status") == "other_issues")
    verif_failed = sum(1 for res in latest_results.values() if res.get("verification_status") == "verification_failed")

    print(f"Total Unique Instances Processed: {total_unique_attempted}")
    print(f"  Generation Succeeded:         {gen_success}")
    print(f"  Repo Setup Failed:            {repo_fail}")
    print(f"  Other Processing Errors:      {proc_error}")
    print("-" * 30)
    verifiable_instances = sum(1 for res in latest_results.values() if res.get("generation_status") == "generation_success" and res.get("verification_status") not in ["skipped_repo_setup_failed", "skipped_processing_error"])
    print(f"Verification Results (for {verifiable_instances} verifiable instances):")
    print(f"  Successfully Reproduced:      {reproduced}")
    print(f"  Resolved Instead:             {resolved_instead}")
    print(f"  Other Issues Printed:         {other_issues}")
    print(f"  Verification Command Failed:  {verif_failed}")
    print("-" * 30)
    if verifiable_instances > 0:
        reproduction_rate = (reproduced / verifiable_instances) * 100
        print(f"Reproduction Rate (Reproduced / Verifiable*): {reproduction_rate:.2f}%")
        print(f"*Verifiable = Gen Succeeded & Repo Setup OK ({verifiable_instances} instances)")
    else: print("Reproduction Rate: N/A (No instances were verifiable)")
    output_folder = os.path.dirname(results_file_path)
    print(f"\nLogs and detailed results saved in: {output_folder}")

if __name__ == "__main__":
    main()