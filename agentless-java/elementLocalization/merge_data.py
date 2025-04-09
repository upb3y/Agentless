#!/usr/bin/env python3
import os
import json
import glob
import argparse

def merge_json_files(directory, output_file, file_pattern="*.json"):
    """
    Merge all JSON files in a directory into a single JSON file.
    
    Args:
        directory: Path to the directory containing JSON files
        output_file: Path to save the merged JSON file
        file_pattern: Pattern to match files (default: "*.json")
    """
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False
    
    # Find all JSON files in the directory
    file_paths = glob.glob(os.path.join(directory, file_pattern))
    if not file_paths:
        print(f"No files matching '{file_pattern}' found in '{directory}'.")
        return False
    
    print(f"Found {len(file_paths)} files to merge.")
    
    # Dictionary to store merged data
    merged_data = {}
    
    # Process each file
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract instance_id from filename
            filename = os.path.basename(file_path)
            if "_related_elements.json" in filename:
                instance_id = filename.replace("_related_elements.json", "")
            else:
                instance_id = filename.replace(".json", "")
            
            # Add to merged data
            merged_data[instance_id] = data
            print(f"Processed: {filename}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing {file_path}: {e}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save merged data
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        print(f"\nSuccessfully merged {len(merged_data)} files into '{output_file}'.")
        return True
    except Exception as e:
        print(f"Error saving merged file: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSON files into a single file")
    parser.add_argument("directory", help="Directory containing JSON files to merge")
    parser.add_argument("--output", "-o", default="merged_.json", help="Output file path (default: merged_results.json)")
    parser.add_argument("--pattern", "-p", default="*.json", help="File pattern to match (default: *.json)")
    
    args = parser.parse_args()
    
    merge_json_files(args.directory, args.output, args.pattern)