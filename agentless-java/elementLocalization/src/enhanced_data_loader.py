import json
import os
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clone_repo(repo_url, base_commit, clone_dir):
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

def load_enhanced_input(issue_file, instance_id=None, clone_repositories=False, clone_dir="./repos"):
    """
    Load data from enhanced JSON input file containing complete issue information.
    Optionally clone the repository at the specified commit.
    
    Args:
        issue_file: Path to the JSON file with issue information
        instance_id: Optional specific instance ID to process
        clone_repositories: Whether to clone the repository
        clone_dir: Directory to clone repositories into
        
    Returns:
        Dictionary with repository, issue description, and suspicious files
    """
    with open(issue_file, 'r') as f:
        data = json.load(f)
    
    # Find the instance with the specified ID
    instance = None
    if instance_id:
        for item in data:
            if item.get('instance_id') == instance_id:
                instance = item
                break
        if not instance:
            raise ValueError(f"No instance found with ID: {instance_id}")
    else:
        # Return the first instance if no specific ID is provided
        instance = data[0]
    
    result = {
        "repo": instance.get('repo'),
        "problem_statement": instance.get('problem_statement'),
        "suspicious_files": get_suspicious_files(instance),
        "repository_structure": instance.get('repository_structure', []),
        "instance_id": instance.get('instance_id'),
        "base_commit": instance.get('base_commit')
    }
    
    # Clone the repository if requested
    if clone_repositories and result["repo"] and result["base_commit"]:
        repo_path = clone_repo(result["repo"], result["base_commit"], clone_dir)
        if repo_path:
            result["repo_path"] = repo_path
        else:
            logger.warning(f"Failed to clone repository {result['repo']}")
    
    return result

def get_suspicious_files(instance):
    """
    Extract suspicious files based on available information.
    Try to use predefined suspicious_files_intersection if available,
    otherwise extract from the instance data itself.
    """
    # First check for suspicious_files_intersection
    if 'suspicious_files_intersection' in instance:
        return instance['suspicious_files_intersection']
    
    # Then check for suspicious_files
    if 'suspicious_files' in instance:
        return instance['suspicious_files']
    
    # Otherwise try to infer from repository_structure
    # This is just a placeholder - you might need a more sophisticated approach
    # based on the actual structure of your data
    return []


def get_file_content(repo_path, file_path):
    """
    Read file content from the cloned repository.
    
    Args:
        repo_path: Path to the cloned repository
        file_path: Relative path to the file within the repository
        
    Returns:
        String content of the file or None if file not found
    """
    full_path = os.path.join(repo_path, file_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, UnicodeDecodeError) as e:
        logger.error(f"Error reading file {full_path}: {e}")
        return None