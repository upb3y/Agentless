#!/usr/bin/env python3
import json
import os
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluation')

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_metrics(test_outputs, ground_truth_data, input_data):
    """
    Calculate binary touch (100% if at least one file is touched) and superset accuracy.
    
    Args:
        test_outputs: Dict of test outputs by instance_id
        ground_truth_data: Ground truth data
        input_data: Input data with file/class/method details
    
    Returns:
        Dictionary of metrics
    """
    results = {}
    
    # Counters for superset accuracy
    file_superset_count = 0
    class_superset_count = 0
    method_superset_count = 0
    
    # Counters for binary touch
    file_touched_count = 0
    class_touched_count = 0
    method_touched_count = 0
    
    total_instances = 0
    
    # Process each instance
    for instance in ground_truth_data:
        instance_id = instance.get("instance_id")
        if not instance_id or instance_id not in test_outputs:
            continue
        
        total_instances += 1
            
        # Get the corresponding ground truth and test output
        gt_files = set()
        gt_classes = set()
        gt_methods = set()
        
        # Extract ground truth locations
        for loc in instance.get("detailed_locations", []):
            file_path = loc.get("file_path")
            class_name = loc.get("class_name", "")
            method_name = loc.get("method_name", "")
            
            if file_path:
                gt_files.add(file_path)
                if class_name:
                    gt_classes.add(f"{file_path}:{class_name}")
                if method_name:
                    gt_methods.add(f"{file_path}:{class_name}:{method_name}")
        
        # Extract test output locations
        test_files = set()
        test_classes = set()
        test_methods = set()
        
        test_output = test_outputs[instance_id]
        for item in test_output:
            file_path = item.get("file_path")
            class_name = item.get("class_name", "")
            method_name = item.get("method_name", "")
            
            if file_path:
                test_files.add(file_path)
                if class_name:
                    test_classes.add(f"{file_path}:{class_name}")
                if method_name:
                    test_methods.add(f"{file_path}:{class_name}:{method_name}")
        
        # Calculate intersection
        file_intersection = gt_files.intersection(test_files)
        class_intersection = gt_classes.intersection(test_classes)
        method_intersection = gt_methods.intersection(test_methods)
        
        # Binary touch metrics (100% if at least one element is touched)
        file_is_touched = len(file_intersection) > 0
        class_is_touched = len(class_intersection) > 0
        method_is_touched = len(method_intersection) > 0
        
        # Increment touched counters
        if file_is_touched:
            file_touched_count += 1
        if class_is_touched:
            class_touched_count += 1
        if method_is_touched:
            method_touched_count += 1
        
        # Original touch percentages (for reference)
        file_touch_percentage = (len(file_intersection) / len(gt_files)) * 100 if gt_files else 0
        class_touch_percentage = (len(class_intersection) / len(gt_classes)) * 100 if gt_classes else 0
        method_touch_percentage = (len(method_intersection) / len(gt_methods)) * 100 if gt_methods else 0
        
        # Binary touch percentages (100% or 0%)
        file_binary_touch = 100 if file_is_touched else 0
        class_binary_touch = 100 if class_is_touched else 0
        method_binary_touch = 100 if method_is_touched else 0
        
        # Calculate superset accuracy (are all ground truth locations contained in test output?)
        is_file_superset = len(file_intersection) == len(gt_files) if gt_files else True
        is_class_superset = len(class_intersection) == len(gt_classes) if gt_classes else True
        is_method_superset = len(method_intersection) == len(gt_methods) if gt_methods else True
        
        # Increment superset counters
        if is_file_superset:
            file_superset_count += 1
        if is_class_superset:
            class_superset_count += 1
        if is_method_superset:
            method_superset_count += 1
        
        # Calculate percentage of test files that are in the input data
        input_files = set()
        for inp in input_data:
            if inp.get("instance_id") == instance_id:
                input_files = set(inp.get("modified_files", []))
                break
        
        # Normalize both file paths for comparison
        normalized_test_files = {normalize_file_paths(path) for path in test_files}
        normalized_input_files = {normalize_file_paths(path) for path in input_files}
        
        # Debug information
        if len(normalized_input_files) > 0:
            logger.debug(f"Normalized input files for {instance_id}: {normalized_input_files}")
            logger.debug(f"Normalized test files for {instance_id}: {normalized_test_files}")
                
        test_in_input = normalized_test_files.intersection(normalized_input_files)
        logger.debug(f"Intersection for {instance_id}: {test_in_input}")
        test_in_input_percentage = (len(test_in_input) / len(test_files)) * 100 if test_files else 0
        
        # Calculate weighted confidence score for test output
        confidence_sum = 0
        confidence_count = 0
        for item in test_output:
            confidence = item.get("confidence", 0)
            if confidence:
                confidence_sum += confidence
                confidence_count += 1
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
        
        results[instance_id] = {
            "file_level": {
                "ground_truth_count": len(gt_files),
                "test_output_count": len(test_files),
                "intersection_count": len(file_intersection),
                "is_touched": file_is_touched,
                "touch_percentage": file_touch_percentage,  # Original proportional touch
                "binary_touch": file_binary_touch,         # Binary touch (100% or 0%)
                "is_superset": is_file_superset,
                "ground_truth_files": list(gt_files),
                "test_output_files": list(test_files),
                "intersection_files": list(file_intersection)
            },
            "class_level": {
                "ground_truth_count": len(gt_classes),
                "test_output_count": len(test_classes),
                "intersection_count": len(class_intersection),
                "is_touched": class_is_touched,
                "touch_percentage": class_touch_percentage,  # Original proportional touch
                "binary_touch": class_binary_touch,         # Binary touch (100% or 0%)
                "is_superset": is_class_superset
            },
            "method_level": {
                "ground_truth_count": len(gt_methods),
                "test_output_count": len(test_methods),
                "intersection_count": len(method_intersection),
                "is_touched": method_is_touched,
                "touch_percentage": method_touch_percentage,  # Original proportional touch
                "binary_touch": method_binary_touch,         # Binary touch (100% or 0%)
                "is_superset": is_method_superset
            },
            "test_in_input": {
                "input_files_count": len(input_files),
                "test_in_input_count": len(test_in_input),
                "test_in_input_percentage": test_in_input_percentage
            },
            "confidence": {
                "average_confidence": avg_confidence
            }
        }
    
    # Calculate binary touch and superset accuracy percentages
    touch_metrics = {
        "file_level_touch_count": file_touched_count,
        "class_level_touch_count": class_touched_count,
        "method_level_touch_count": method_touched_count,
        "file_level_touch_rate": (file_touched_count / total_instances) * 100 if total_instances else 0,
        "class_level_touch_rate": (class_touched_count / total_instances) * 100 if total_instances else 0,
        "method_level_touch_rate": (method_touched_count / total_instances) * 100 if total_instances else 0,
    }
    
    superset_metrics = {
        "file_level_superset_accuracy": (file_superset_count / total_instances) * 100 if total_instances else 0,
        "class_level_superset_accuracy": (class_superset_count / total_instances) * 100 if total_instances else 0,
        "method_level_superset_accuracy": (method_superset_count / total_instances) * 100 if total_instances else 0,
        "file_superset_count": file_superset_count,
        "class_superset_count": class_superset_count,
        "method_superset_count": method_superset_count,
        "total_instances": total_instances
    }
    
    return results, touch_metrics, superset_metrics

def calculate_average_metrics(results):
    """Calculate average metrics across all instances."""
    if not results:
        return {}
        
    avg_metrics = {
        "file_level_touch_percentage": 0,  # Original proportional touch
        "file_level_binary_touch": 0,     # Binary touch
        "class_level_touch_percentage": 0,
        "class_level_binary_touch": 0,
        "method_level_touch_percentage": 0,
        "method_level_binary_touch": 0,
        "test_in_input_percentage": 0,
        "average_confidence": 0
    }
    
    for instance_id, metrics in results.items():
        avg_metrics["file_level_touch_percentage"] += metrics["file_level"]["touch_percentage"]
        avg_metrics["file_level_binary_touch"] += metrics["file_level"]["binary_touch"]
        avg_metrics["class_level_touch_percentage"] += metrics["class_level"]["touch_percentage"]
        avg_metrics["class_level_binary_touch"] += metrics["class_level"]["binary_touch"]
        avg_metrics["method_level_touch_percentage"] += metrics["method_level"]["touch_percentage"]
        avg_metrics["method_level_binary_touch"] += metrics["method_level"]["binary_touch"]
        avg_metrics["test_in_input_percentage"] += metrics["test_in_input"]["test_in_input_percentage"]
        avg_metrics["average_confidence"] += metrics.get("confidence", {}).get("average_confidence", 0)
    
    num_instances = len(results)
    for key in avg_metrics:
        avg_metrics[key] /= num_instances
        
    return avg_metrics

def process_test_outputs(test_output_dir):
    """
    Process individual test output files in a directory and merge them.
    
    Args:
        test_output_dir: Directory containing individual test output files
        
    Returns:
        Dictionary of merged test outputs by instance_id
    """
    merged_outputs = {}
    
    if not os.path.isdir(test_output_dir):
        print(f"Test output directory {test_output_dir} not found.")
        return merged_outputs
    
    for filename in os.listdir(test_output_dir):
        if not filename.endswith('_related_elements.json'):
            continue
        
        instance_id = filename.split('_related_elements.json')[0]
        file_path = os.path.join(test_output_dir, filename)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                merged_outputs[instance_id] = data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return merged_outputs

def normalize_file_paths(file_path):
    """Normalize file paths to handle common variations."""
    if not file_path:
        return ""
        
    # Remove leading slash if present
    if file_path.startswith('/'):
        file_path = file_path[1:]
    
    # Handle src/main vs src/main/java variations
    if '/src/main/' in file_path and '/java/' not in file_path:
        file_path = file_path.replace('/src/main/', '/src/main/java/')
    
    # Remove 'src/' prefix if present
    if file_path.startswith('src/'):
        file_path = file_path[4:]
    
    # Handle main/java vs java variations
    if '/main/java/' in file_path:
        alt_path = file_path.replace('/main/java/', '/java/')
        return alt_path
    
    # Normalize path separators
    file_path = file_path.replace('\\', '/')
    
    # Convert to lowercase for case-insensitive comparison
    file_path = file_path.lower()
    
    return file_path

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate localization effectiveness with binary touch metric')
    parser.add_argument('--merged_output_file', help='Path to merged test outputs file')
    parser.add_argument('--test_output_dir', default='test_output', help='Directory containing individual test output files')
    parser.add_argument('--input_data_file', default="merged_suspicious_files_with_data.json", help='Path to input data file (used for metadata)')
    parser.add_argument('--ground_truth_file', default="output/ground_truth.json", help='Path to ground truth file')
    parser.add_argument('--output_file', default="localization_evaluation_results.json", help='Path to save evaluation results')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load ground truth data (mandatory)
    ground_truth_data = load_json_file(args.ground_truth_file)
    if not ground_truth_data:
        print(f"Failed to load ground truth data from {args.ground_truth_file}. Exiting.")
        return
    
    print(f"Loaded {len(ground_truth_data)} ground truth instances")
    
    # Load input data (optional - used for additional metadata)
    input_data = load_json_file(args.input_data_file) or []
    if not input_data:
        print(f"Warning: Failed to load input data from {args.input_data_file}.")
        print("Continuing with only ground truth data.")
    
    # Load test outputs (either from merged file or directory)
    test_outputs = {}
    if args.merged_output_file:
        test_outputs = load_json_file(args.merged_output_file)
    else:
        test_outputs = process_test_outputs(args.test_output_dir)
    
    if not test_outputs:
        print("No test outputs found. Exiting.")
        return
    
    print(f"Loaded test outputs for {len(test_outputs)} instances")
    
    # Calculate metrics
    results, touch_metrics, superset_metrics = calculate_metrics(test_outputs, ground_truth_data, input_data)
    if not results:
        print("No results generated. Exiting.")
        return
    
    print(f"Generated evaluation results for {len(results)} instances")
    
    # Calculate average metrics
    avg_metrics = calculate_average_metrics(results)
    
    # Print results
    print("\n=== Localization Effectiveness Evaluation ===\n")
    
    print("Binary Touch Metrics (100% if any file/class/method is touched):")
    print(f"  File-level touch rate: {touch_metrics['file_level_touch_rate']:.2f}% ({touch_metrics['file_level_touch_count']}/{superset_metrics['total_instances']})")
    print(f"  Class-level touch rate: {touch_metrics['class_level_touch_rate']:.2f}% ({touch_metrics['class_level_touch_count']}/{superset_metrics['total_instances']})")
    print(f"  Method-level touch rate: {touch_metrics['method_level_touch_rate']:.2f}% ({touch_metrics['method_level_touch_count']}/{superset_metrics['total_instances']})")
    
    print("\nAverage Binary Touch Percentages:")
    print(f"  File-level binary touch: {avg_metrics['file_level_binary_touch']:.2f}%")
    print(f"  Class-level binary touch: {avg_metrics['class_level_binary_touch']:.2f}%")
    print(f"  Method-level binary touch: {avg_metrics['method_level_binary_touch']:.2f}%")
    
    print("\nAverage Original Touch Percentages (for reference):")
    print(f"  File-level touch percentage: {avg_metrics['file_level_touch_percentage']:.2f}%")
    print(f"  Class-level touch percentage: {avg_metrics['class_level_touch_percentage']:.2f}%")
    print(f"  Method-level touch percentage: {avg_metrics['method_level_touch_percentage']:.2f}%")
    
    print(f"\nOther Metrics:")
    print(f"  Test output files in input data: {avg_metrics['test_in_input_percentage']:.2f}%")
    if avg_metrics['test_in_input_percentage'] == 0:
        print("  WARNING: Test in input percentage is 0%. This suggests file path normalization issues or missing input data.")
    print(f"  Average confidence score: {avg_metrics['average_confidence']:.2f}/5")
    
    print("\nSuperset Accuracy (Test output contains all ground truth locations):")
    print(f"  File-level superset accuracy: {superset_metrics['file_level_superset_accuracy']:.2f}% ({superset_metrics['file_superset_count']}/{superset_metrics['total_instances']})")
    print(f"  Class-level superset accuracy: {superset_metrics['class_level_superset_accuracy']:.2f}% ({superset_metrics['class_superset_count']}/{superset_metrics['total_instances']})")
    print(f"  Method-level superset accuracy: {superset_metrics['method_level_superset_accuracy']:.2f}% ({superset_metrics['method_superset_count']}/{superset_metrics['total_instances']})")
    
    print("\nDetailed Results by Instance:")
    for instance_id, metrics in results.items():
        print(f"\nInstance: {instance_id}")
        print(f"  File-level binary touch: {'100%' if metrics['file_level']['is_touched'] else '0%'}")
        print(f"  File-level original touch: {metrics['file_level']['touch_percentage']:.2f}% ({metrics['file_level']['intersection_count']}/{metrics['file_level']['ground_truth_count']})")
        print(f"  File-level superset: {'Yes' if metrics['file_level']['is_superset'] else 'No'}")
        
        print(f"  Class-level binary touch: {'100%' if metrics['class_level']['is_touched'] else '0%'}")
        print(f"  Class-level original touch: {metrics['class_level']['touch_percentage']:.2f}% ({metrics['class_level']['intersection_count']}/{metrics['class_level']['ground_truth_count']})")
        print(f"  Class-level superset: {'Yes' if metrics['class_level']['is_superset'] else 'No'}")
        
        print(f"  Method-level binary touch: {'100%' if metrics['method_level']['is_touched'] else '0%'}")
        print(f"  Method-level original touch: {metrics['method_level']['touch_percentage']:.2f}% ({metrics['method_level']['intersection_count']}/{metrics['method_level']['ground_truth_count']})")
        print(f"  Method-level superset: {'Yes' if metrics['method_level']['is_superset'] else 'No'}")
        
        print(f"  Average confidence: {metrics['confidence']['average_confidence']:.2f}/5")
        
        # Print detailed file lists if needed
        # Comment out these sections if you prefer a more concise output
        print("  Ground Truth Files:")
        for file_path in metrics['file_level']['ground_truth_files']:
            print(f"    - {file_path}")
            
        print("  Test Output Files:")
        for file_path in metrics['file_level']['test_output_files']:
            print(f"    - {file_path}")
        
        print("  Intersection Files:")
        for file_path in metrics['file_level']['intersection_files']:
            print(f"    - {file_path}")
    
    # Save results to a JSON file
    with open(args.output_file, 'w') as f:
        json.dump({
            "average_metrics": avg_metrics,
            "touch_metrics": touch_metrics,
            "superset_metrics": superset_metrics,
            "instance_results": results
        }, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()