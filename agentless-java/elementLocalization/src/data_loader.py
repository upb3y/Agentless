# src/data_loader.py
import pandas as pd
import os
import json

def load_swe_bench_java_data(row_index=None):
    """
    Load SWE-bench-Java data from the HuggingFace dataset.
    
    Args:
        row_index: Optional index to load a specific example
        
    Returns:
        A dictionary with issue description and suspicious files
    """
    # Load dataset
    df = pd.read_json("hf://datasets/Daoguang/Multi-SWE-bench/swe-bench-java-verified.json")
    
    # Select specific row or first row by default
    row = df.iloc[row_index if row_index is not None else 0]
    
    # Extract repo and issue information
    repo = row['repo']
    problem_statement = row['problem_statement']
    patch = row['patch']
    
    # Extract modified files from the patch
    modified_files = extract_files_from_patch(patch)
    
    return {
        "repo": repo,
        "issue_description": problem_statement,
        "suspicious_files": modified_files,
        "patch": patch
    }

def extract_files_from_patch(patch):
    """
    Extract modified file paths from a git patch.
    """
    files = []
    for line in patch.split('\n'):
        if line.startswith('diff --git '):
            # Extract the path after 'b/' (second file path in the diff)
            parts = line.split(' ')
            if len(parts) >= 4:
                file_path = parts[3][2:]  # Remove 'b/' prefix
                files.append(file_path)
    
    return files