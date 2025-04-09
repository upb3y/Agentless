#!/usr/bin/env python3
import json
import argparse

def add_problem_statements_and_commits(full_repository_file, merged_suspicious_files, output_file):
    """
    Add problem statements and base_commit from full_repository.json to merged_suspicious_files.json
    
    Args:
        full_repository_file: Path to the full repository JSON file with problem statements and base commits
        merged_suspicious_files: Path to the merged suspicious files JSON
        output_file: Path to save the output JSON with problem statements and base commits added
    """
    # Load full repository data
    try:
        with open(full_repository_file, 'r') as f:
            full_repository = json.load(f)
    except Exception as e:
        print(f"Error loading full repository file: {e}")
        return False
    
    # Create lookup dictionary for problem statements and base_commits by instance_id
    problem_data = {}
    for repo in full_repository:
        instance_id = repo.get("instance_id")
        if instance_id:
            problem_data[instance_id] = {
                "problem_statement": repo.get("problem_statement", ""),
                "base_commit": repo.get("base_commit", "")
            }
    
    print(f"Loaded data for {len(problem_data)} instances")
    
    # Load merged suspicious files
    try:
        with open(merged_suspicious_files, 'r') as f:
            suspicious_files = json.load(f)
    except Exception as e:
        print(f"Error loading merged suspicious files: {e}")
        return False
    
    # Add problem statements and base_commits to merged suspicious files
    count_added = 0
    count_missing = 0
    for item in suspicious_files:
        instance_id = item.get("instance_id")
        if instance_id and instance_id in problem_data:
            item["problem_statement"] = problem_data[instance_id]["problem_statement"]
            item["base_commit"] = problem_data[instance_id]["base_commit"]
            count_added += 1
        else:
            count_missing += 1
            print(f"Warning: No data found for instance {instance_id}")
    
    # Save the updated data
    try:
        with open(output_file, 'w') as f:
            json.dump(suspicious_files, f, indent=2)
        print(f"Successfully added data for {count_added} instances")
        if count_missing > 0:
            print(f"Missing data for {count_missing} instances")
        print(f"Updated data saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving output file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Add problem statements and base commits to merged suspicious files")
    parser.add_argument("--full_repo", default="full_repository_structure.json", help="Path to full repository JSON file")
    parser.add_argument("--merged_files", default="merged_suspicious_files.json", help="Path to merged suspicious files JSON")
    parser.add_argument("--output", default="merged_suspicious_files_with_data.json", help="Path to save the output JSON")
    
    args = parser.parse_args()
    
    add_problem_statements_and_commits(args.full_repo, args.merged_files, args.output)

if __name__ == "__main__":
    main()