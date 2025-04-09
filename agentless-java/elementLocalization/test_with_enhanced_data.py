# test_with_enhanced_data.py
import os
import json
import argparse
from src.enhanced_data_loader import load_enhanced_input, get_file_content
from src.localization import localize_related_elements
from src.llm_client import MockLLMClient, OpenAIClient, GeminiClient
from src.java_parser import extract_skeleton, format_skeleton

def test_with_enhanced_data(input_file, instance_id, llm="mock", api_key=None, model="gpt-3.5-turbo", clone_dir="repos"):
    """
    Test the AGENTLESS Step 4 process using enhanced input file.
    Uses dynamic repository cloning instead of static repository paths.
    """
    # Load data from enhanced input and clone the repository
    data = load_enhanced_input(
        issue_file=input_file, 
        instance_id=instance_id,
        clone_repositories=True,
        clone_dir=clone_dir
    )
    
    print(f"Processing instance: {instance_id}")
    print(f"Repository: {data['repo']}")
    print(f"Base commit: {data.get('base_commit', 'Not specified')}")
    print(f"Issue description: {data['problem_statement'][:100]}...")
    
    # Check if repository was successfully cloned
    if 'repo_path' not in data:
        print("Repository not successfully cloned. Exiting.")
        return None
    
    repo_path = data['repo_path']
    print(f"Using repository at: {repo_path}")
    
    # Initialize LLM client
    if llm == "openai" and api_key:
        llm_client = OpenAIClient(api_key, model)
    elif llm == "gemini" and api_key:
        llm_client = GeminiClient(api_key)
    else:
        llm_client = MockLLMClient()
    
    # Run localization
    related_elements = localize_related_elements(
        repo_path,
        data["suspicious_files"],
        data["problem_statement"],
        llm_client
    )
    
    # Save the results
    os.makedirs("test_output", exist_ok=True)
    output_file = f"test_output/{instance_id}_related_elements.json"
    with open(output_file, 'w') as f:
        json.dump(related_elements, f, indent=2)
    
    print(f"Localization completed. Found {len(related_elements)} related elements.")
    print(f"Results saved to {output_file}")
    
    # Also output the results to console
    print("\nLocalized Elements:")
    for elem in related_elements:
        print(f"- {elem['file_path']} | {elem['class_name']} | {elem['method_name']}")
        print(f"  Reason: {elem['reason']}")
    
    return related_elements

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AGENTLESS Step 4 with enhanced input")
    parser.add_argument("--input_file", required=True, help="Path to merged input JSON file")
    parser.add_argument("--instance_id", required=True, help="Specific instance ID to process")
    parser.add_argument("--llm", choices=["mock", "openai", "gemini"], default="mock", help="LLM to use")
    parser.add_argument("--api_key", help="API key for the chosen LLM")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model to use with OpenAI")
    parser.add_argument("--stdout_json", default="false", help="Output JSON to stdout instead of file")
    parser.add_argument("--clone_dir", default="repos", help="Directory to clone repositories into")

    args = parser.parse_args()
    
    # Call the function and get results
    result_data = test_with_enhanced_data(
        args.input_file, 
        args.instance_id,
        args.llm, 
        args.api_key, 
        args.model,
        args.clone_dir
    )
    
    # After getting the result, decide how to output it
    if result_data is not None:
        if args.stdout_json.lower() == "true":
            # Instead of writing to file, print to stdout
            print(json.dumps(result_data))
        else:
            # File was already written in the function, this is redundant but kept for clarity
            output_file = f"test_output/{args.instance_id}_related_elements.json"
            print(f"Results saved to {output_file}")