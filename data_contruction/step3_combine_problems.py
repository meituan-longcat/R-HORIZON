import json
import re
import os
import random

def extract_boxed_answer(text):
    """Extract content from the last \\boxed{} in text"""
    if not text:
        return None
    
    # Find all \boxed{} patterns
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last match
        return matches[-1].strip()
    
    return None

def get_target_integer(target_list):
    """Extract integer from target list"""
    for target in target_list:
        if is_pure_integer(target):
            try:
                cleaned = re.sub(r'[\\{}\s]', '', target)
                return int(cleaned)
            except ValueError as e:
                print(f"Warning: Could not convert '{target}' -> '{cleaned}' to integer: {e}")
                continue
    return None

def is_pure_integer(text):
    """Check if text is a pure integer"""
    # Check for LaTeX commands
    if re.search(r'\\[a-zA-Z]', text):
        return False

    # Check for letters
    if re.search(r'[a-zA-Z]', text):
        return False

    # Check for fractions
    if '/' in text:
        return False

    # Check for decimal points
    if '.' in text:
        return False

    # Remove spaces, braces, etc.
    cleaned = re.sub(r'[\\{}\s]', '', text)

    # Final check for pure integer
    pattern = r'^-?\d+$'
    return bool(re.match(pattern, cleaned))

def create_dependency_expression(prev_answer_placeholder, prev_answer_value, selected_number):
    """Create dependency expression ensuring variable equals selected_number"""
    # Calculate required difference: selected_number = prev_answer_value + diff
    diff = selected_number - prev_answer_value

    if diff >= 0:
        return f"{prev_answer_placeholder} + {diff}"
    else:
        return f"{prev_answer_placeholder} - {abs(diff)}"

def create_linked_input(prev_answer_placeholder, prev_answer_value, sample, problem_index):
    """Create linked input by replacing selected variable"""
    
    input_text = sample.get('input', '')
    selected_variable = sample.get('selected_variable', {})
    
    # Get selected variable information
    selected_number = selected_variable.get('number')
    start_pos = selected_variable.get('start_pos')
    end_pos = selected_variable.get('end_pos')
    is_in_math_env = selected_variable.get('is_in_math_env', False)
    
    if selected_number is None or start_pos is None or end_pos is None:
        print(f"Warning: Sample {sample.get('instanceId', 'unknown')} missing selected_variable information")
        return None

    # Create variable name based on problem index and math environment
    base_var_name = f"[variable{problem_index}]"
    if is_in_math_env:
        # In math environment, use braces to ensure variable is treated as a whole
        var_name = f"{{{base_var_name}}}"
    else:
        var_name = base_var_name

    # Create dependency expression
    dependency_expr = create_dependency_expression(prev_answer_placeholder, prev_answer_value, selected_number)

    # Construct linked input
    linked_input = f"Using the result {prev_answer_placeholder} from the previous calculation, {base_var_name} = {dependency_expr}. "

    # Replace selected number in original problem
    modified_input = input_text[:start_pos] + var_name + input_text[end_pos:]
    linked_input += modified_input

    return linked_input

def load_key_variable_samples(input_file):
    """Load samples with key variable information"""
    samples = []
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found - {input_file}")
        return samples
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read().strip()
        
        # Split by double newlines to get each JSON object
        json_blocks = content.split('\n\n')
        
        for i, block in enumerate(json_blocks):
            block = block.strip()
            if not block:
                continue
                
            try:
                sample = json.loads(block)
                
                # Verify required fields
                if all(key in sample for key in ['instanceId', 'input', 'target', 'selected_variable']):
                    samples.append(sample)
                else:
                    print(f"Warning: Sample {i+1} missing required fields")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON block {i+1}: {e}")
                continue
    
    print(f"Successfully loaded {len(samples)} key variable samples")
    return samples

def create_single_combination(samples_group, k, combination_id):
    """Create a single combined sample"""
    # Validate all samples have valid integer targets
    valid_group = True
    target_values = []  # Store target values for each problem
    original_targets = []  # Store original targets
    group_targets = []  # Store array of targets in order

    for sample in samples_group:
        target_val = get_target_integer(sample['target'])
        if target_val is None:
            valid_group = False
            break

        target_values.append(target_val)
        original_targets.append(sample['target'])
        group_targets.append(sample['target'])  # Add to group_targets array

    if not valid_group:
        return None

    # Create combined input parts
    combined_input_parts = []
    instance_ids = []
    selected_variables = []  # Record all selected variables

    for j, sample in enumerate(samples_group):
        instance_ids.append(sample.get("instanceId", f"unknown_{combination_id}_{j}"))
        selected_variables.append(sample.get("selected_variable", {}))

        if j == 0:
            # First sample uses original input directly
            combined_input_parts.append(f"Problem {j+1}: {sample['input']}")
        else:
            # Subsequent samples link to previous answer using [] notation
            prev_answer_placeholder = f"[answer{j}]"  # [answer1], [answer2], etc.
            prev_answer_value = target_values[j-1]  # Target value of previous problem

            # Create linked input with selected variable information
            linked_input = create_linked_input(
                prev_answer_placeholder,
                prev_answer_value,
                sample,
                j + 1  # Problem number (second problem is 2, etc.)
            )

            # Skip if linked input creation failed
            if linked_input is None:
                return None

            combined_input_parts.append(f"Problem {j+1}: {linked_input}")

    # Combine into complete input
    combined_input = "\n\n".join(combined_input_parts)
    
    # Build answer format example with [] notation
    answer_format_lines = []
    for i in range(1, k + 1):
        answer_format_lines.append(f"Problem {i}: \\boxed{{[answer{i}]}}")
    
    answer_format = "\n\n".join(answer_format_lines)
    
    # Add final instructions and format requirements with notation explanation
    combined_input += f"""\n\nNote: In this problem set:
- [variablek] represents the calculated variable needed to solve problem k.
- [answerk] represents the answer to problem k.

Solve all problems step by step and provide the answers for all problems in the following format:

### Final Answers

{answer_format}
"""
    # Create combined target - join all answers into one string
    target_strings = []
    for target in original_targets:
        if isinstance(target, list) and len(target) > 0:
            target_strings.append(str(target[0]))  # Take first element
        else:
            target_strings.append(str(target))
    
    combined_target_string = ",".join(target_strings)
    combined_target = [combined_target_string]  # Wrap in list

    # Create combined sample maintaining original dataset format
    combined_example = {
        "input": combined_input,
        "instanceId": instance_ids[-1],  # Use last problem's instanceId
        "target": combined_target,
        # Additional fields
        "instanceIds": instance_ids,
        "num_problems": k,
        "problem_type": "chained_reasoning",
        "group_targets": group_targets,  # Array of each problem's target in order
        "selected_variables": selected_variables,  # Store all selected variables
    }

    # Preserve other fields from last sample if they exist
    last_sample = samples_group[-1]
    for key, value in last_sample.items():
        if key not in combined_example:  # Don't overwrite existing fields
            combined_example[key] = value

    return combined_example


def combine_samples_for_k_fixed_num(key_variable_samples, output_dir, input_filename, k, seed, target_num_samples=500):
    """Combine samples for specified k value, generating fixed number of samples"""

    if len(key_variable_samples) < k:
        print(f"Warning: Not enough key variable samples ({len(key_variable_samples)}) for k={k}, skipping")
        return None, []

    # Build output file path
    output_filename = f"combined_key_var_k{k}_sd{seed}_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)

    # Shuffle sample order
    shuffled_samples = key_variable_samples.copy()
    random.shuffle(shuffled_samples)

    all_combined_examples = []  # Store all combined samples
    skipped_no_replacement = 0  # Count combinations skipped due to replacement issues

    # Use set to store generated combinations for efficient deduplication
    generated_combinations = set()

    print(f"k={k}: Combining with key variables, target {target_num_samples} samples")

    # Phase 1: Random sampling to generate samples
    attempt_count = 0
    max_attempts = target_num_samples * 5

    while len(all_combined_examples) < target_num_samples and attempt_count < max_attempts:
        # Randomly select k samples from pool
        if len(shuffled_samples) >= k:
            random_group = random.sample(shuffled_samples, k)

            # Check if this combination has already been generated
            current_instance_ids = tuple(sample.get("instanceId", "") for sample in random_group)

            if current_instance_ids in generated_combinations:
                attempt_count += 1
                continue

            combined_example = create_single_combination(random_group, k, len(all_combined_examples))
            if combined_example is not None:
                all_combined_examples.append(combined_example)
                generated_combinations.add(current_instance_ids)
            else:
                skipped_no_replacement += 1

        attempt_count += 1

    print(f"k={k}: Generated {len(all_combined_examples)} samples in random sampling phase")

    # Phase 2: If random sampling is insufficient, use permutations to supplement
    if len(all_combined_examples) < target_num_samples:
        print(f"k={k}: Insufficient random samples, starting permutation supplement phase")

        max_direct_combinations = len(shuffled_samples) // k

        # Generate base combinations
        base_groups = []
        for i in range(max_direct_combinations):
            start_idx = i * k
            group = shuffled_samples[start_idx:start_idx + k]
            base_groups.append(group)

        # Generate more samples through permutations
        combination_count = len(all_combined_examples)
        retry_count = 0
        max_retries = (target_num_samples - len(all_combined_examples)) * 3

        while len(all_combined_examples) < target_num_samples and retry_count < max_retries:
            if not base_groups:
                break

            # Randomly select a base combination
            base_group = random.choice(base_groups)

            # Shuffle the order of samples in this combination
            shuffled_group = base_group.copy()
            random.shuffle(shuffled_group)

            # Check if this combination has already been generated
            current_instance_ids = tuple(sample.get("instanceId", "") for sample in shuffled_group)

            if current_instance_ids in generated_combinations:
                retry_count += 1
                continue

            combined_example = create_single_combination(shuffled_group, k, combination_count)
            if combined_example is None:
                skipped_no_replacement += 1
                retry_count += 1
                continue

            all_combined_examples.append(combined_example)
            generated_combinations.add(current_instance_ids)
            combination_count += 1
            retry_count = 0  # Reset retry count

        print(f"k={k}: Supplemented samples in permutation phase")

    # Truncate to target number
    all_combined_examples = all_combined_examples[:target_num_samples]

    print(f"k={k}: Final generated {len(all_combined_examples)} samples, skipped {skipped_no_replacement}")

    # Write to file, maintaining original dataset format
    with open(output_path, 'w', encoding='utf-8') as outfile:
        output_data = {
            "examples": all_combined_examples,
            "statistics": {
                "num_combined_samples": len(all_combined_examples),
                "k_value": k,
                "target_num_samples": target_num_samples,
                "skipped_no_replacement": skipped_no_replacement,
                "using_key_variables": True
            }
        }
        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

    print(f"k={k}: Saved {len(all_combined_examples)} samples to {output_path}")

    return output_path, all_combined_examples

def create_pretty_files(output_paths, output_dir):
    """Create formatted, human-readable versions of all output files"""
    pretty_paths = []

    for output_path in output_paths:
        if output_path is None:
            continue

        # Extract k value
        filename = os.path.basename(output_path)
        k_match = re.search(r'k(\d+)', filename)
        if k_match:
            k = k_match.group(1)
            pretty_filename = f"combined_key_var_k{k}_pretty_formatted.jsonl"
        else:
            pretty_filename = f"pretty_{filename}"

        pretty_output_path = os.path.join(output_dir, pretty_filename)

        with open(output_path, 'r', encoding='utf-8') as infile, \
             open(pretty_output_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                data = json.loads(line.strip())
                outfile.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

        pretty_paths.append(pretty_output_path)

    return pretty_paths

def main():
    # Path configuration
    key_variable_input_path = "./data_contruction/data/key_variables_results.jsonl"
    output_dir = "./data_contruction/data/combined_samples"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set different k values and target sample count
    k_values = [2, 3, 4, 5]
    target_num_samples = 30  # Target number of samples for each k value

    # Check if input file exists
    if not os.path.exists(key_variable_input_path):
        print(f"Error: Input file not found - {key_variable_input_path}")
        return

    # Set random seed for reproducibility
    seed = 43
    random.seed(seed)

    # Load key variable samples
    print("Loading key variable samples...")
    key_variable_samples = load_key_variable_samples(key_variable_input_path)

    if not key_variable_samples:
        print("No key variable samples found")
        return

    input_filename = os.path.basename(key_variable_input_path)
    output_paths = []
    k_combined_samples = {}  # Store combined samples for each k value

    print(f"\nStarting to generate combined data for different k values...")
    print(f"k values: {k_values}")
    print(f"Available key variable samples: {len(key_variable_samples)}")
    print(f"Target samples per k value: {target_num_samples}")

    # Generate data for each k value
    for k in k_values:
        print(f"\nProcessing k={k}...")
        output_path, combined_samples = combine_samples_for_k_fixed_num(
            key_variable_samples, output_dir, input_filename, k, seed, target_num_samples
        )
        if output_path:
            output_paths.append(output_path)
            k_combined_samples[k] = combined_samples

    # Create formatted, human-readable files
    print(f"\nCreating pretty-formatted files...")
    pretty_paths = create_pretty_files(output_paths, output_dir)

    # Output summary
    print(f"\n" + "="*60)
    print(f"Processing complete! Generated {len(output_paths)} data files:")
    for i, (k, output_path) in enumerate(zip(k_values, output_paths)):
        if output_path:
            print(f"k={k:2d}: {os.path.basename(output_path)}")
        else:
            print(f"k={k:2d}: Skipped (insufficient samples)")

    print(f"\nPretty-formatted files:")
    for pretty_path in pretty_paths:
        print(f"       {os.path.basename(pretty_path)}")

    print(f"\n" + "="*80)
    print(f"Combined Sample Statistics (target samples: {target_num_samples}):")
    print(f"{'k value':<10} {'Actual samples':<15}")
    print("-" * 80)

    # Add statistics for each k value
    for k in k_values:
        if k in k_combined_samples and k_combined_samples[k]:
            print(f"k={k:<7} {len(k_combined_samples[k]):<15}")
        else:
            print(f"k={k:<7} 0")

    print(f"\nVariable notation explanation:")
    print(f"- [variablek]: Calculated variable needed to solve problem k")
    print(f"- [answerk]: Answer to problem k")
    print(f"- Variables in math environments are marked with {{[variablek]}}")
    print(f"\nAll files saved to: {output_dir}")

if __name__ == "__main__":
    main()
    