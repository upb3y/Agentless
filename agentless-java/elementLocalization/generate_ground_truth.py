# enhanced_ground_truth_generator.py
import pandas as pd
import os
import json
import re
import argparse
import subprocess
import logging
import shutil
import tempfile
from datasets import load_dataset
from typing import Dict, List, Set, Tuple, Optional, Any
import javalang  # You might need to pip install javalang

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ground_truth_generator')

def extract_files_from_patch(patch):
    """
    Extract modified file paths from a git patch.
    Using the exact same implementation as data_loader.py
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

def extract_detailed_locations_from_patch(patch):
    """
    Extract file, class, and method level information from a git patch.
    Returns a list of dictionaries with file_path, class_name, and method_name.
    """
    if not patch:
        return []
        
    locations = []
    current_file = None
    current_class = None
    current_method = None
    
    lines = patch.split('\n')
    for i, line in enumerate(lines):
        # Extract file path from diff line
        if line.startswith('diff --git '):
            parts = line.split(' ')
            if len(parts) >= 4:
                current_file = parts[3][2:]  # Remove 'b/' prefix
        
        # Check if we're in a file section
        if not current_file:
            continue
            
        # Look for class and method definitions in context
        if line.startswith('@@') and i < len(lines) - 1:
            # Reset class and method when we see a new hunk
            current_class = None
            current_method = None
            
            # Look ahead in the context to find class and method names
            context_start = i + 1
            context_end = min(i + 20, len(lines))  # Look at next 20 lines as context
            
            for j in range(context_start, context_end):
                if j >= len(lines):
                    break
                    
                context_line = lines[j]
                
                # Skip diff markers for contextual analysis
                if context_line.startswith('+') or context_line.startswith('-'):
                    context_line = context_line[1:]
                
                # Very simplified Java parsing - would need to be improved for real use
                class_match = re.search(r'class\s+([A-Za-z0-9_]+)', context_line)
                if class_match:
                    current_class = class_match.group(1)
                
                # Look for method definitions
                method_match = re.search(r'(?:@\w+\s+)*(?:public|private|protected|static|final|native|synchronized|abstract|transient)?\s+(?:[A-Za-z0-9_.<>[\],\s]+)\s+([A-Za-z0-9_]+)\s*\(', context_line)
                if method_match:
                    current_method = method_match.group(1)
        
        # For modified lines with + or -
        if line.startswith('+') or line.startswith('-'):
            # Create a location entry if we haven't already for this file/class/method combination
            location = {
                'file_path': current_file,
                'class_name': current_class,
                'method_name': current_method
            }
            
            # Filter out None values
            location = {k: v for k, v in location.items() if v is not None}
                
            # Check if this location is already in our list to avoid duplicates
            if location not in locations:
                locations.append(location)
    
    return locations

def extract_added_lines_from_patch(patch_text):
    """
    Parses a unified diff and returns a dictionary:
    {
      file_path: [list of added line numbers]
    }
    """
    file_to_lines = {}
    patch_blocks = patch_text.split("diff --git ")
    for block in patch_blocks:
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        # Find the file path (from the diff --git line)
        if not lines:
            continue
        header = lines[0]  # e.g., "a/foo.java b/foo.java"
        match = re.match(r"a\/(.+?)\s+b\/(.+)", header)
        if not match:
            continue
        file_path = match.group(2).strip()  # Get 'b/...' path
        # Process hunks
        hunk_blocks = re.split(r"(@@.*?@@)", block)
        for i in range(1, len(hunk_blocks), 2):
            hunk_header = hunk_blocks[i]
            hunk_body = hunk_blocks[i + 1] if i + 1 < len(hunk_blocks) else ""
            hunk_match = re.search(r"\+(\d+)(?:,(\d+))?", hunk_header)
            if not hunk_match:
                continue
            start_line = int(hunk_match.group(1))
            current_line = start_line
            for line in hunk_body.splitlines():
                if line.startswith('+') and not line.startswith('+++'):
                    file_to_lines.setdefault(file_path, []).append(current_line)
                    current_line += 1
                elif line.startswith(' ') or (line and not line.startswith('-')):
                    current_line += 1
                # Removed lines do not increment line number
    return file_to_lines

def parse_java_content(file_content):
    """
    Parse Java file content to create mappings between line numbers, classes, and methods.
    
    Args:
        file_content: The content of the Java file as a string
        
    Returns:
        A tuple containing:
        - A dictionary mapping line numbers to class names
        - A dictionary mapping line numbers to method names
        - A dictionary mapping class names to their line ranges
        - A dictionary mapping (class_name, method_name) to their line ranges
    """
    try:
        # Parse the Java content using javalang
        tree = javalang.parse.parse(file_content)
        
        # Initialize mappings
        line_to_class = {}
        line_to_method = {}
        class_ranges = {}
        method_ranges = {}
        
        # Get positions of all tokens to calculate line numbers
        tokens = list(javalang.tokenizer.tokenize(file_content))
        
        # Helper function to get line number for a position
        def find_line_number(position):
            for i, token in enumerate(tokens):
                if token.position >= position:
                    return token.position[0]
            return len(file_content.splitlines())
        
        # Helper function to find end position
        def find_end_position(start_position, body_content):
            if not body_content:
                return start_position
            
            # Count braces to find matching end
            body_str = file_content[start_position:]
            open_braces = 0
            for i, char in enumerate(body_str):
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        return start_position + i
            
            return len(file_content)
        
        # Process classes
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = node.name
            # Find the starting line number for this class
            class_start = find_line_number(node.position)
            
            # Find approximate end position
            class_end = find_end_position(node.position[1], node.body)
            class_end_line = find_line_number((0, class_end))
            
            # Store class range
            class_ranges[class_name] = (class_start, class_end_line)
            
            # Map all lines in this class
            for line in range(class_start, class_end_line + 1):
                line_to_class[line] = class_name
            
            # Process methods in this class
            for method_path, method_node in node.filter(javalang.tree.MethodDeclaration):
                method_name = method_node.name
                
                # Find the starting line number for this method
                method_start = find_line_number(method_node.position)
                
                # Find approximate end position
                method_end = find_end_position(method_node.position[1], method_node.body)
                method_end_line = find_line_number((0, method_end))
                
                # Store method range
                method_key = (class_name, method_name)
                method_ranges[method_key] = (method_start, method_end_line)
                
                # Map all lines in this method
                for line in range(method_start, method_end_line + 1):
                    line_to_method[line] = method_name
        
        return line_to_class, line_to_method, class_ranges, method_ranges
        
    except Exception as e:
        print(f"Error parsing Java content: {e}")
        return {}, {}, {}, {}

def find_class_method_for_line(repo_path, file_path, line_number):
    """
    Find the class and method containing a specific line in a Java file.
    
    Args:
        repo_path: Path to the repository
        file_path: Path to the Java file (relative to repo_path)
        line_number: Line number to find class/method for
        
    Returns:
        Tuple of (class_name, method_name) or (None, None) if not found
    """
    # Try to read the file - skip if not found or not accessible
    try:
        full_path = os.path.join(repo_path, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None
    
    # Skip non-Java files
    if not file_path.endswith('.java'):
        return None, None
    
    # Parse Java content to find mappings
    line_to_class, line_to_method, _, _ = parse_java_content(content)
    
    # Find class and method for the line
    class_name = line_to_class.get(line_number)
    method_name = line_to_method.get(line_number)
    
    return class_name, method_name

def extract_detailed_locations_with_lines(patch, repo_path=None):
    """
    Extract file, class, method, and line level information from a git patch.
    If repo_path is provided, it will attempt to parse the Java files to find
    more precise class and method information for each line.
    
    Returns:
        List of dictionaries with file_path, class_name, method_name, and line_number.
    """
    if not patch:
        return []
    
    # Get basic locations from the patch (file, class, method)
    basic_locations = extract_detailed_locations_from_patch(patch)
    
    # Get line numbers for each file
    file_to_lines = extract_added_lines_from_patch(patch)
    
    # If no repo_path provided, we can only associate lines with existing locations
    if not repo_path:
        # Add line numbers to existing locations when possible
        enhanced_locations = []
        for location in basic_locations:
            file_path = location.get('file_path')
            if file_path in file_to_lines:
                for line_number in file_to_lines[file_path]:
                    # Create a new location entry with line number
                    new_loc = location.copy()
                    new_loc['line_number'] = line_number
                    enhanced_locations.append(new_loc)
            else:
                # Keep the original location if no line numbers
                enhanced_locations.append(location)
        
        return enhanced_locations
    
    # With repo_path, we can do more precise analysis
    enhanced_locations = []
    
    # Process each file with its added lines
    for file_path, lines in file_to_lines.items():
        for line_number in lines:
            # Try to find class and method for this line
            class_name, method_name = find_class_method_for_line(repo_path, file_path, line_number)
            
            # Create a location entry
            location = {
                'file_path': file_path,
                'line_number': line_number
            }
            
            # Add class and method if found
            if class_name:
                location['class_name'] = class_name
            if method_name:
                location['method_name'] = method_name
                
            # Add to our enhanced locations
            enhanced_locations.append(location)
    
    # If no enhanced locations, fall back to basic ones
    if not enhanced_locations:
        return basic_locations
    
    return enhanced_locations

def clone_repo(repo_url, base_commit, clone_dir):
    """
    Clones a repository at a specific commit.
    
    Args:
        repo_url: Repository URL in format 'org/repo'
        base_commit: Git commit hash to checkout
        clone_dir: Directory to clone into
    
    Returns:
        Path to the cloned repository or None if failed
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

def generate_enhanced_ground_truth(output_file, repo_base_path=None, dynamic_cloning=False, clone_temp_dir=None, limit=None):
    """
    Generate an enhanced ground truth JSON file with file, class, method, and line level information.
    
    Args:
        output_file: Path to save the consolidated ground truth JSON file
        repo_base_path: Optional base path to local copies of repositories for deeper analysis
        dynamic_cloning: Whether to dynamically clone repositories if not found locally
        clone_temp_dir: Directory to clone repositories into (if dynamic_cloning is True)
        limit: Optional limit on number of instances to process (for testing)
    """
    # Create a temporary directory for cloning if needed
    temp_dir = None
    if dynamic_cloning and not clone_temp_dir:
        temp_dir = tempfile.mkdtemp()
        clone_temp_dir = temp_dir
        logger.info(f"Created temporary directory for cloning: {clone_temp_dir}")
    elif dynamic_cloning:
        os.makedirs(clone_temp_dir, exist_ok=True)
        logger.info(f"Using directory for cloning: {clone_temp_dir}")
    try:
        print("Loading dataset...")
        # Try loading directly with datasets if available
        try:
            dataset = load_dataset("Daoguang/Multi-SWE-bench")
            df = pd.DataFrame(dataset["java_verified"])
            print(f"Successfully loaded dataset with {len(df)} instances using datasets")
        except Exception as e:
            print(f"Failed to load with datasets: {e}")
            # Fallback to direct loading
            df = pd.read_json("hf://datasets/Daoguang/Multi-SWE-bench/swe-bench-java-verified.json")
            print(f"Successfully loaded dataset with {len(df)} instances using direct path")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have the datasets library installed and configured correctly.")
        print("If you're having authentication issues, make sure you're logged in to Hugging Face:")
        print("  - Run: huggingface-cli login")
        return
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        df = df.head(limit)
        print(f"Limited processing to first {limit} instances")
    
    # Create enhanced ground truth data
    ground_truth_data = []
    
    # Track statistics
    total_instances = len(df)
    instances_with_files = 0
    instances_with_classes = 0
    instances_with_methods = 0
    instances_with_lines = 0
    
    # Process each instance
    for index, row in df.iterrows():
        # Extract information
        repo = row['repo']
        problem_statement = row['problem_statement']
        patch = row['patch']
        
        # Generate instance_id from repository name if not provided
        instance_id = row.get('instance_id')
        if not instance_id:
            repo_name = repo.split('/')[-1] if isinstance(repo, str) else f"instance-{index}"
            instance_id = f"{repo_name}-{index}"
        
        # Extract file paths
        modified_files = extract_files_from_patch(patch)
        if modified_files:
            instances_with_files += 1
        
        # Get base commit (needed for dynamic cloning)
        base_commit = row.get('base_commit', None)
        
        # Find repository path for this instance
        instance_repo_path = None
        
        # First check if it exists in repo_base_path
        if repo_base_path and isinstance(repo, str):
            repo_name = repo.split('/')[-1]
            possible_path = os.path.join(repo_base_path, repo_name)
            if os.path.exists(possible_path):
                instance_repo_path = possible_path
                logger.info(f"Found existing repository for {repo} at {possible_path}")
        
        # If not found and dynamic_cloning is enabled, try to clone it
        if not instance_repo_path and dynamic_cloning and isinstance(repo, str) and base_commit:
            logger.info(f"Repository {repo} not found locally, attempting to clone...")
            instance_repo_path = clone_repo(repo, base_commit, clone_temp_dir)
            if instance_repo_path:
                logger.info(f"Successfully cloned {repo} at commit {base_commit}")
            else:
                logger.warning(f"Failed to clone repository {repo}")
        
        if instance_repo_path:
            logger.info(f"Using repository path: {instance_repo_path} for analysis")
        
        # Extract enhanced detailed locations with line numbers
        detailed_locations = extract_detailed_locations_with_lines(patch, instance_repo_path)
        
        # Count statistics
        has_classes = any('class_name' in loc for loc in detailed_locations)
        has_methods = any('method_name' in loc for loc in detailed_locations)
        has_lines = any('line_number' in loc for loc in detailed_locations)
        
        if has_classes:
            instances_with_classes += 1
        if has_methods:
            instances_with_methods += 1
        if has_lines:
            instances_with_lines += 1
        
        # Count unique elements
        unique_classes = set(loc.get('class_name') for loc in detailed_locations if 'class_name' in loc)
        unique_methods = set((loc.get('class_name', ''), loc.get('method_name', '')) 
                            for loc in detailed_locations if 'method_name' in loc)
        unique_lines = set((loc.get('file_path', ''), loc.get('line_number', -1)) 
                          for loc in detailed_locations if 'line_number' in loc)
        
        # Create ground truth entry
        ground_truth_entry = {
            "instance_id": instance_id,
            "repository": repo,
            "problem_statement": problem_statement,
            "modified_files": modified_files,
            "detailed_locations": detailed_locations,
            "stats": {
                "file_count": len(modified_files),
                "class_count": len(unique_classes),
                "method_count": len(unique_methods),
                "line_count": len(unique_lines)
            }
        }
        
        ground_truth_data.append(ground_truth_entry)
        print(f"Processed instance {index} ({instance_id}): {len(modified_files)} files, " +
              f"{len(unique_classes)} classes, {len(unique_methods)} methods, {len(unique_lines)} lines")
    
    # Save consolidated file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(ground_truth_data, f, indent=2)
    
    # Print summary
    print(f"\nEnhanced ground truth file generated with {total_instances} instances")
    print(f"Instances with files: {instances_with_files}/{total_instances} ({100*instances_with_files/total_instances:.1f}%)")
    print(f"Instances with classes: {instances_with_classes}/{total_instances} ({100*instances_with_classes/total_instances:.1f}%)")
    print(f"Instances with methods: {instances_with_methods}/{total_instances} ({100*instances_with_methods/total_instances:.1f}%)")
    print(f"Instances with lines: {instances_with_lines}/{total_instances} ({100*instances_with_lines/total_instances:.1f}%)")
    print(f"Output file: {output_file}")
    
    # Clean up temporary directory if we created one
    if temp_dir and os.path.exists(temp_dir):
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced ground truth file for Java SWE-bench")
    parser.add_argument("--output_file", required=True,
                        help="Path to save the enhanced ground truth JSON file")
    parser.add_argument("--repo_base_path", default=None,
                        help="Base path to local copies of repositories for deeper analysis")
    parser.add_argument("--dynamic_cloning", action="store_true",
                        help="Dynamically clone repositories if not found locally")
    parser.add_argument("--clone_temp_dir", default=None,
                        help="Directory to clone repositories into (if dynamic_cloning is True)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit processing to the first N instances (for testing)")
    
    args = parser.parse_args()
    generate_enhanced_ground_truth(
        args.output_file, 
        args.repo_base_path,
        args.dynamic_cloning,
        args.clone_temp_dir,
        args.limit
    )

if __name__ == "__main__":
    main()