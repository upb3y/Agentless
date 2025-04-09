# driver.py
import os
import json
import argparse
from src.data_loader import load_swe_bench_java_data
from src.localization import localize_related_elements
from src.llm_client import MockLLMClient, OpenAIClient, GeminiClient

def main():
    parser = argparse.ArgumentParser(description="Run AGENTLESS Step 4 for Java")
    parser.add_argument("--repo_path", help="Path to the Java repository")
    parser.add_argument("--row_index", type=int, default=None, help="Index of the SWE-bench-Java row to process")
    parser.add_argument("--llm", choices=["mock", "openai", "gemini"], default="mock", help="LLM to use")
    parser.add_argument("--api_key", help="API key for the chosen LLM")
    parser.add_argument("--output_file", default="results/related_elements.json", help="Path to save the output")
    args = parser.parse_args()
    
    # Load SWE-bench-Java data
    data = load_swe_bench_java_data(args.row_index)
    
    # Initialize LLM client
    if args.llm == "openai" and args.api_key:
        llm_client = OpenAIClient(args.api_key)
    elif args.llm == "gemini" and args.api_key:
        llm_client = GeminiClient(args.api_key)
    else:
        llm_client = MockLLMClient()
    
    # Set repository path
    repo_path = args.repo_path or os.path.join("repos", data["repo"])
    
    # Run localization
    related_elements = localize_related_elements(
        repo_path,
        data["suspicious_files"],
        data["issue_description"],
        llm_client
    )
    
    # Save the results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(related_elements, f, indent=2)
    
    print(f"Localization completed. Found {len(related_elements)} related elements.")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()