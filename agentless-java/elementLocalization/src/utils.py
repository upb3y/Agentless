# utils.py
import os
import json
from typing import List, Dict, Any

def load_suspicious_files(repo_path: str, file_paths: List[str]) -> List[Dict]:
    """
    Load the content of suspicious files.
    
    Args:
        repo_path: Path to the repository
        file_paths: Paths to suspicious files (relative to repo_path)
    
    Returns:
        List of dictionaries with file paths and content
    """
    suspicious_files = []
    
    for rel_path in file_paths:
        abs_path = os.path.join(repo_path, rel_path)
        
        if not os.path.exists(abs_path):
            print(f"Warning: File {abs_path} does not exist")
            continue
        
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            suspicious_files.append({
                'path': rel_path,
                'content': content
            })
        except Exception as e:
            print(f"Error loading file {abs_path}: {e}")
    
    return suspicious_files