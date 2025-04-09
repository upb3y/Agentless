# src/localization.py
import os
import json
from typing import List, Dict, Any
from .java_parser import extract_skeleton, format_skeleton

def localize_related_elements(repo_path, suspicious_files, issue_description, llm_client):
    """
    Step 4 of AGENTLESS: Localize to related elements within suspicious files.
    Process files in smaller batches to avoid token limits.
    
    Args:
        repo_path: Path to the Java repository
        suspicious_files: List of suspicious file paths
        issue_description: The original issue description
        llm_client: A client to query the LLM
    
    Returns:
        List of related classes and functions
    """
    # Filter for Java files only
    java_files = [f for f in suspicious_files if f.endswith('.java')]
    
    # Process files in batches
    batch_size = 15  # Adjust based on your average file size
    all_related_elements = []
    
    print(f"Processing {len(java_files)} Java files in batches of {batch_size}")
    
    for i in range(0, len(java_files), batch_size):
        batch_files = java_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: {batch_files}")
        
        # 1. Generate skeletons for the current batch
        file_skeletons = []
        for file_path in batch_files:
            skeleton_data = extract_skeleton(repo_path, file_path)
            formatted_skeleton = format_skeleton(skeleton_data)
            file_skeletons.append({
                'path': file_path,
                'skeleton': formatted_skeleton,
                'skeleton_data': skeleton_data
            })
        
        # 2. Combine skeletons for the current batch
        batch_skeletons = "\n\n".join([fs['skeleton'] for fs in file_skeletons])
        
        # 3. Create a prompt for the current batch
        prompt = f"""
        You are a Java expert helping diagnose and fix an issue in a Java repository.

        THE ISSUE:
        {issue_description}

        RELEVANT JAVA FILES STRUCTURE (BATCH {i//batch_size + 1} OF {(len(java_files) + batch_size - 1) // batch_size}):
        Below are the skeletons (class and method declarations) of some potentially relevant files:

        {batch_skeletons}

        YOUR TASK:
        Based on the issue description and Java code structures above, identify ONLY the most likely elements (classes and methods) that would need to be MODIFIED to fix this issue. Be selective and precise.

        CONSTRAINTS:
        - Select a MAXIMUM of 2-3 methods per class
        - Focus ONLY on methods that would need to be modified to fix the issue
        - Include only highly relevant elements with clear connection to the issue
        - Do not include methods that are merely related but wouldn't need modification

        RESPOND ONLY with a valid JSON array containing objects with these fields:
        - file_path: The path to the file
        - class_name: The name of the class (can be empty for interface issues)
        - method_name: The name of the method (can be empty for class-level issues)
        - reason: Brief explanation of why this specific element would need modification to fix the issue
        - confidence: A number from 1-5 where 5 means you're very confident this element needs modification

        EXAMPLE RESPONSE:
        ```json
        [
        {{
            "file_path": "src/main/java/org/example/User.java",
            "class_name": "User",
            "method_name": "authenticate",
            "reason": "This method has a logic error that directly causes the authentication failure described in the issue",
            "confidence": 5
        }},
        {{
            "file_path": "src/main/java/org/example/AuthUtils.java",
            "class_name": "AuthUtils",
            "method_name": "validateToken",
            "reason": "This method doesn't properly handle expired tokens as mentioned in the issue",
            "confidence": 4
        }}
        ]
        """
        
        try:
            # 4. Query the LLM for the current batch
            llm_response = llm_client.query(prompt)
            
            # 5. Parse the response
            batch_related_elements = extract_json_from_response(llm_response)
            
            # Filter to only include higher confidence results
            filtered_elements = [element for element in batch_related_elements if element.get('confidence', 0) >= 3]
            all_related_elements.extend(filtered_elements)
            print(f"Found {len(batch_related_elements)} related elements, kept {len(filtered_elements)} with confidence >= 3 in batch {i//batch_size + 1}")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
    
    # Deduplicate elements based on file_path, class_name, and method_name
    unique_elements = {}
    for element in all_related_elements:
        key = (element.get('file_path', ''), element.get('class_name', ''), element.get('method_name', ''))
        if key not in unique_elements:
            unique_elements[key] = element
    
    deduplicated_elements = list(unique_elements.values())
    print(f"Total unique related elements found: {len(deduplicated_elements)}")
    
    return deduplicated_elements

def extract_json_from_response(response):
    """
    Extract JSON from the LLM response with better error handling.
    """
    import re
    import json
    
    # First try to find JSON array with relaxed pattern
    json_matches = re.findall(r'\[\s*\{.+?\}\s*\]', response, re.DOTALL)
    for json_match in json_matches:
        try:
            return json.loads(json_match)
        except json.JSONDecodeError:
            continue
    
    # If that fails, look for code blocks with JSON
    code_block_pattern = r'```(?:json)?\s*(\[\s*\{.+?\}\s*\])```'
    code_matches = re.findall(code_block_pattern, response, re.DOTALL)
    for code_match in code_matches:
        try:
            return json.loads(code_match)
        except json.JSONDecodeError:
            continue
    
    # Try to find any JSON array in the response
    try:
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
    except Exception as e:
        print(f"Failed to extract JSON with approach 3: {e}")
    
    # If all else fails, try a line-by-line approach to find valid JSON
    lines = response.split('\n')
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            try:
                json_str = ''.join(lines[i:j+1])
                if '[' in json_str and ']' in json_str:
                    result = json.loads(json_str)
                    if isinstance(result, list) and len(result) > 0:
                        return result
            except:
                continue
    
    print("WARNING: Could not extract JSON from response")
    return []