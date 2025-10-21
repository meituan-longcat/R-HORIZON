import json
import re
import os
from glob import glob

def contains_numbers(text):
    """Check if text contains any numbers"""
    return bool(re.search(r'\d', text))

def contains_integers(text):
    """Check if text contains complete integers (not part of floating-point numbers)"""
    integers = extract_numbers_from_text(text)
    return len(integers) > 0

def is_pure_integer(text):
    """Check if text is a pure integer"""
    if re.search(r'\\[a-zA-Z]', text):  # Contains LaTeX commands
        return False

    if re.search(r'[a-zA-Z]', text):  # Contains letters
        return False

    if '/' in text:  # Contains fractions
        return False

    if '.' in text:  # Contains decimal points
        return False

    # Remove spaces, braces, etc.
    cleaned = re.sub(r'[\\{}\s]', '', text)

    # Final check for pure integer
    pattern = r'^-?\d+$'
    return bool(re.match(pattern, cleaned))

def extract_numbers_from_text(text):
    """Extract all complete integers from text, excluding parts of floating-point numbers"""
    numbers_with_pos = []
    float_ranges = set()

    # Identify floating-point number ranges
    float_patterns = [
        r'\d+\.\d+',  # 123.45
        r'\d+\.',     # 123.
        r'\.\d+',     # .45
    ]

    for pattern in float_patterns:
        for match in re.finditer(pattern, text):
            for pos in range(match.start(), match.end()):
                float_ranges.add(pos)

    # Find valid integers
    integer_pattern = r'\d+'
    for match in re.finditer(integer_pattern, text):
        start_pos = match.start()
        end_pos = match.end()

        # Check if part of a float
        is_part_of_float = any(pos in float_ranges for pos in range(start_pos, end_pos))
        if is_part_of_float:
            continue

        # Check surrounding characters
        before_char = text[start_pos-1] if start_pos > 0 else ' '
        after_char = text[end_pos] if end_pos < len(text) else ' '

        if before_char == '.' or after_char == '.':
            continue

        if (start_pos > 0 and before_char.isalnum() and 
            before_char not in ['$', '£', '€', '¥', '#', ' ', '(', '[', '{']):
            continue

        if end_pos < len(text) and after_char.isalpha():
            continue

        num = int(match.group())
        if num > 0:  # Keep positive integers only
            numbers_with_pos.append({
                'number': num,
                'start': start_pos,
                'end': end_pos,
                'text': match.group()
            })

    return numbers_with_pos

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

def get_filtered_samples(input_path, filtered_output_dir):
    """Filter and process samples from JSON Lines file"""
    filtered_samples = []
    total_samples = 0
    rejected_no_integers = 0

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                # Parse each line as a separate sample
                example = json.loads(line.strip())
                total_samples += 1
                
                input_text = example.get("input", "")
                target = example.get("target", [])

                if not isinstance(target, list):
                    target = [str(target)]

                # Check filtering conditions
                if contains_integers(input_text):
                    all_targets_integer = True
                    for t in target:
                        if not is_pure_integer(str(t)):
                            all_targets_integer = False
                            break

                    if all_targets_integer:
                        filtered_samples.append(example)
                else:
                    rejected_no_integers += 1

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                continue
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

    print(f"\n=== Filtering Results ===")
    print(f"Total samples: {total_samples}")
    print(f"Filtered qualified samples: {len(filtered_samples)}")
    print(f"Samples rejected (no valid integers): {rejected_no_integers}")

    # Save filtered samples in JSON Lines format
    os.makedirs(filtered_output_dir, exist_ok=True)
    filtered_output_path = os.path.join(filtered_output_dir, "filtered_samples.jsonl")

    with open(filtered_output_path, 'w', encoding='utf-8') as outfile:
        # Add statistics as a comment line
        outfile.write(f"# Filtering Stats: Total={total_samples}, Filtered={len(filtered_samples)}, Rejected={rejected_no_integers}\n")
        
        # Write each sample as a separate line
        for sample in filtered_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Filtered samples saved to: {filtered_output_path}")
    return filtered_samples

def main():
    # Path configuration (relative paths for open source)
    input_path = "./data_contruction/data/input_samples.jsonl"  # Input file path
    filtered_output_dir = "./data_contruction/data"  # Output directory

    # Create output directory if it doesn't exist
    os.makedirs(filtered_output_dir, exist_ok=True)

    # Validate input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found - {input_path}")
        return

    # Run filtering process
    print("Starting sample filtering...")
    filtered_samples = get_filtered_samples(input_path, filtered_output_dir)

    if not filtered_samples:
        print("No qualified samples found")
        return

    print(f"\nFiltering completed! {len(filtered_samples)} qualified samples saved")
    print(f"Output directory: {filtered_output_dir}")

if __name__ == "__main__":
    main()
