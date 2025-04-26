#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install google-generativeai


# In[16]:


import os
import json
import re  # Make sure to add this import!
import time
import openai
import subprocess
from pathlib import Path
from datasets import load_dataset
import google.generativeai as genai


# In[3]:


# 📌 Load the dataset
ds = load_dataset("Daoguang/Multi-SWE-bench")
java_dataset = ds["java_verified"]


# In[4]:


# 📌 Define a function to clone a repository at a specific commit
def clone_repo(repo_url, base_commit, clone_dir):
    """Clones a repository at a specific commit."""
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)

    repo_name = repo_url.split("/")[-1]  # Extract repo name
    repo_path = os.path.join(clone_dir, repo_name)

    if os.path.exists(repo_path):
        print(f"✅ Repo {repo_name} already exists, skipping clone...")
    else:
        print(f"🔄 Cloning {repo_url} at commit {base_commit}...")
        subprocess.run(["git", "clone", f"https://github.com/{repo_url}.git", repo_path])
    
    # Checkout the base commit to match dataset state
    subprocess.run(["git", "-C", repo_path, "checkout", base_commit])

    return repo_path


# In[6]:


def create_structure(directory_path):
    """Create a flat tree-like list of .java files and README.md under the repo."""
    structure = []

    for root, dirs, files in os.walk(directory_path):
        # Ignore hidden folders like .git
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        rel_root = os.path.relpath(root, directory_path)
        if rel_root == ".":
            rel_root = ""

        for file_name in sorted(files):
            if file_name.endswith(".java") or file_name.lower() == "readme.md":
                full_path = os.path.join(rel_root, file_name) if rel_root else file_name
                structure.append(full_path)

    return sorted(structure)


# In[ ]:


# 📌 Process each instance in the dataset
all_results = []
CLONE_DIR = "cloned_repos"

for example in java_dataset:
    repo = example["repo"]
    instance_id = example["instance_id"]
    base_commit = example["base_commit"]
    problem_statement = example["problem_statement"]

    print(f"🔍 Processing {repo} - {instance_id}")

    # Use your existing clone logic
    repo_path = clone_repo(repo, base_commit, CLONE_DIR)

    # Create structured file-level dictionary
    structure_dict = create_structure(repo_path)

    # Save result
    instance_result = {
        "repo": repo,
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "base_commit": base_commit,
        "repository_structure": structure_dict
    }
    all_results.append(instance_result)


# In[12]:


# 📌 Save to JSON
with open("full_repository_structure.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("✅ All repository structures saved in full_repository_structure.json")


# In[13]:


# 📌 Load repository structure data
with open("full_repository_structure.json", "r") as f:
    repo_data = json.load(f)


# In[20]:


# 🔹 Replace with your Gemini API key
GOOGLE_API_KEY = "AIzaSyBSVrinqp7ouHqCk456WahVyILewNkViXM"

# Configure the API key
genai.configure(api_key=GOOGLE_API_KEY)


# In[21]:


def extract_json_from_text(raw_text):
    if not raw_text:
        return "[]"
    try:
        json.loads(raw_text)
        return raw_text
    except:
        pass
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    code_blocks = re.findall(code_block_pattern, raw_text)
    for block in code_blocks:
        try:
            json.loads(block.strip())
            return block.strip()
        except:
            pass
    array_pattern = r'\[([\s\S]*?)\]'
    match = re.search(array_pattern, raw_text)
    if match:
        try:
            json_array = f"[{match.group(1)}]"
            json.loads(json_array)
            return json_array
        except:
            pass
    file_paths = re.findall(r'"([^"]+\.[a-zA-Z0-9]+)"', raw_text)
    if file_paths:
        return json.dumps(file_paths)
    return "[]"


# In[22]:


def identify_suspicious_files_chunked(issue_description, repo_structure_list, chunk_size=300, top_n=50):
    all_candidates = []
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192
        }
    )

    for i in range(0, len(repo_structure_list), chunk_size):
        chunk = repo_structure_list[i:i+chunk_size]
        chunk_text = "\n".join(chunk)
        prompt = f"""
You are an expert software engineer skilled in debugging large codebases.
Your task is to analyze the given issue description and this PARTIAL repository structure 
to identify files that are potentially relevant to the issue.

### Issue Description ###
{issue_description}

### Partial Repository Structure ###
{chunk_text}

### Task ###
Return a JSON list of suspicious file paths that are likely relevant to the issue.
If none seem relevant, return an empty list.
Format your response as:
["path/to/file1.java", "path/to/file2.java", ...]
        """

        try:
            response = model.generate_content(prompt)
            raw_output = response.text.strip()
            json_str = extract_json_from_text(raw_output)
            json_output = json.loads(json_str)
            all_candidates.extend(json_output)
        except Exception as e:
            print(f"❌ Error processing chunk {i//chunk_size+1}: {e}")
            time.sleep(60)

        time.sleep(30)

    unique_files = list(dict.fromkeys(all_candidates))  # deduplicate
    return unique_files[:top_n]


# In[23]:


with open("598_project/full_repository_structure.json", "r") as f:
    repo_data = json.load(f)

all_suspicious_files = []
for instance in repo_data:
    repo = instance["repo"]
    instance_id = instance["instance_id"]
    issue_description = instance["problem_statement"]
    repo_structure = instance["repository_structure"]

    print(f"🔍 Processing {repo} - {instance_id}...")

    suspicious_files = identify_suspicious_files_chunked(
        issue_description, repo_structure, chunk_size=300, top_n=50
    )

    all_suspicious_files.append({
        "repo": repo,
        "instance_id": instance_id,
        "suspicious_files": suspicious_files
    })

    with open("suspicious_files.json", "w") as f:
        json.dump(all_suspicious_files, f, indent=2)


# In[24]:


all_suspicious_files


# In[25]:


# 🔹 Save results to JSON file
output_file = "suspicious_files_results.json"
with open(output_file, "w") as f:
    json.dump(all_suspicious_files, f, indent=4)

print(f"✅ Processing completed! Results saved in {output_file}")
