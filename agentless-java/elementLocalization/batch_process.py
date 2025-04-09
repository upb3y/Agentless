# batch_process.py
import json
import subprocess
import os
import argparse
import time

def process_instances(input_file, output_file="combined_results.json", instance_ids=None, llm="mock", api_key=None, model="gpt-3.5-turbo", force=False):
    """
    Process multiple instances from the input file and save all results to a single output file.
    
    Args:
        input_file: Path to the merged data JSON file
        output_file: Path to save the combined results
        instance_ids: List of specific instance IDs to process (None means process all)
        llm: LLM to use ("mock", "openai", etc.)
        api_key: API key for the chosen LLM
        model: Model to use with OpenAI
        force: Whether to force reprocessing of instances with existing results
    """
    # Load the merged data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract all instance IDs if none provided
    if not instance_ids:
        instance_ids = []
        for instance in data:
            if isinstance(instance, dict) and 'instance_id' in instance:
                instance_ids.append(instance['instance_id'])
    
    print(f"Found {len(instance_ids)} instances to process.")
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Try to load existing combined results if they exist
    combined_results = {}
    if os.path.exists(output_file) and not force:
        try:
            with open(output_file, 'r') as f:
                combined_results = json.load(f)
            print(f"Loaded existing results from {output_file}")
        except json.JSONDecodeError:
            print(f"Error loading existing results from {output_file}. Starting fresh.")
    
    # Process each instance
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, instance_id in enumerate(instance_ids):
        print(f"\nProcessing [{idx+1}/{len(instance_ids)}] {instance_id}...")
        
        # Skip if results already exist in combined file and not forced
        if instance_id in combined_results and not force:
            print(f"Results already exist for {instance_id}, skipping.")
            skipped += 1
            continue
        
        # Build command
        cmd = [
            "python", "test_with_enhanced_data.py",
            "--input_file", input_file,
            "--instance_id", instance_id,
            "--llm", llm
        ]
        
        if api_key and llm != "mock":
            cmd.extend(["--api_key", api_key])
        
        if model and llm == "openai":
            cmd.extend(["--model", model])
        
        # Add output option to get JSON on stdout
        cmd.extend(["--stdout_json", "true"])
        
        # Run the command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Try to parse the JSON output
            try:
                result_json = json.loads(result.stdout.strip())
                # Add to combined results
                combined_results[instance_id] = result_json
                
                # Save after each successful processing to avoid losing data
                with open(output_file, 'w') as f:
                    json.dump(combined_results, f, indent=2)
                
                successful += 1
                print(f"Successfully processed {instance_id}")
            except json.JSONDecodeError as je:
                failed += 1
                print(f"Error parsing JSON output for {instance_id}: {je}")
                print(f"Raw output: {result.stdout}")
            
            # Add delay between API calls to avoid rate limits
            if llm != "mock" and idx < len(instance_ids) - 1:
                print(f"Waiting 2 seconds before next instance...")
                time.sleep(2)
                
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"Error processing {instance_id}: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            
            # If API error, wait longer before the next request
            if llm != "mock" and idx < len(instance_ids) - 1:
                print(f"API error detected. Waiting 10 seconds before next instance...")
                time.sleep(10)
    
    print(f"\nProcessing complete! Successful: {successful}, Failed: {failed}, Skipped: {skipped}")
    print(f"Combined results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process multiple instances")
    parser.add_argument("--input_file", required=True, help="Path to merged JSON file")
    parser.add_argument("--output_file", default="combined_results.json", help="Path to save combined results")
    parser.add_argument("--llm", choices=["mock", "openai", "gemini"], default="mock", help="LLM to use")
    parser.add_argument("--api_key", help="API key for the LLM")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model to use with OpenAI")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of instances with existing results")
    parser.add_argument("--ids", help="Comma-separated list of specific instance IDs to process")
    
    args = parser.parse_args()
    
    # Process specific IDs or all instances
    if args.ids:
        specific_ids = [id.strip() for id in args.ids.split(',')]
        process_instances(args.input_file, args.output_file, specific_ids, args.llm, args.api_key, args.model, args.force)
    else:
        process_instances(args.input_file, args.output_file, None, args.llm, args.api_key, args.model, args.force)