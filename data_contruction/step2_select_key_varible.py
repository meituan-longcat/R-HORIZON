import json
import os
import requests
import re
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Tuple
import random

def is_in_complex_expression(text, start_pos, end_pos):
    """Check if integer is in a complex expression (cannot be replaced independently)"""
    
    # Expand check range to see more context
    context_start = max(0, start_pos - 30)
    context_end = min(len(text), end_pos + 30)
    context = text[context_start:context_end]
    
    # Calculate relative position in context
    relative_start = start_pos - context_start
    relative_end = end_pos - context_start
    
    # 1. Check if in common mathematical functions
    math_functions = [
        r'sqrt\s*\(',
        r'sin\s*\(',
        r'cos\s*\(',
        r'tan\s*\(',
        r'log\s*\(',
        r'ln\s*\(',
        r'exp\s*\(',
        r'abs\s*\(',
        r'floor\s*\(',
        r'ceil\s*\(',
        r'round\s*\(',
        r'max\s*\(',
        r'min\s*\(',
        r'pow\s*\(',
        r'mod\s*\(',
        r'gcd\s*\(',
        r'lcm\s*\('
    ]
    
    for func_pattern in math_functions:
        for match in re.finditer(func_pattern, context, re.IGNORECASE):
            func_start = match.start()
            # Find corresponding closing parenthesis
            paren_count = 0
            func_end = -1
            for i in range(match.end() - 1, len(context)):
                if context[i] == '(':
                    paren_count += 1
                elif context[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        func_end = i + 1
                        break
            
            # Check if integer is inside this function
            if func_end > 0 and func_start < relative_start and relative_end < func_end:
                return True, f"Inside function: {match.group(0)}"
    
    # 2. Check LaTeX mathematical functions
    latex_functions = [
        r'\\sqrt\s*\{',
        r'\\frac\s*\{',
        r'\\sin\s*\{',
        r'\\cos\s*\{',
        r'\\tan\s*\{',
        r'\\log\s*\{',
        r'\\ln\s*\{',
        r'\\exp\s*\{',
        r'\\binom\s*\{',
        r'\\sum\s*_\{',
        r'\\prod\s*_\{',
        r'\\int\s*_\{'
    ]
    
    for latex_pattern in latex_functions:
        for match in re.finditer(latex_pattern, context, re.IGNORECASE):
            func_start = match.start()
            # Find corresponding closing brace
            brace_count = 0
            func_end = -1
            for i in range(match.end() - 1, len(context)):
                if context[i] == '{':
                    brace_count += 1
                elif context[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        func_end = i + 1
                        break
            
            # Check if integer is inside this LaTeX function
            if func_end > 0 and func_start < relative_start and relative_end < func_end:
                return True, f"Inside LaTeX function: {match.group(0)}"
    
    # 3. Check if in fraction expressions
    # Check fractions like a/b
    fraction_pattern = r'(\d+)\s*/\s*(\d+)'
    for match in re.finditer(fraction_pattern, context):
        if match.start() <= relative_start and relative_end <= match.end():
            return True, f"Part of fraction: {match.group(0)}"
    
    # Check LaTeX fractions \frac{a}{b}
    latex_frac_pattern = r'\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}'
    for match in re.finditer(latex_frac_pattern, context):
        if match.start() <= relative_start and relative_end <= match.end():
            return True, f"Part of LaTeX fraction: {match.group(0)}"
    
    # 4. Check if in exponent expressions
    # Check exponents like a^b
    exponent_patterns = [
        r'(\d+)\s*\^\s*(\d+)',
        r'(\d+)\s*\*\*\s*(\d+)',
        r'(\w+)\s*\^\s*(\d+)',
        r'(\w+)\s*\*\*\s*(\d+)'
    ]
    
    for exp_pattern in exponent_patterns:
        for match in re.finditer(exp_pattern, context):
            if match.start() <= relative_start and relative_end <= match.end():
                return True, f"Part of exponent: {match.group(0)}"
    
    return False, ""

def is_in_latex_math_environment(text, start_pos, end_pos):
    """Check if integer is in a LaTeX math environment"""
    
    # Get larger context to detect math environments
    context_start = max(0, start_pos - 300)
    context_end = min(len(text), end_pos + 300)
    context = text[context_start:context_end]
    
    # Calculate relative position in context
    relative_start = start_pos - context_start
    relative_end = end_pos - context_start
    
    # 1. Check inline math environment $...$
    inline_math_pattern = r'\$([^$]*?)\$'
    for match in re.finditer(inline_math_pattern, context):
        math_start = match.start()
        math_end = match.end()
        if math_start < relative_start and relative_end < math_end:
            return True, f"Inside inline math: ${match.group(1)[:20]}..."
    
    # 2. Check display math environment $$...$$
    display_math_pattern = r'\$\$([^$]*?)\$\$'
    for match in re.finditer(display_math_pattern, context, re.DOTALL):
        math_start = match.start()
        math_end = match.end()
        if math_start < relative_start and relative_end < math_end:
            return True, f"Inside display math: $${match.group(1)[:20]}..."
    
    # 3. Check equation environments \[...\] and \\[...\\]
    equation_patterns = [
        r'\\\[(.*?)\\\]',      # Single backslash \[...\]
        r'\\\\\\[(.*?)\\\\\\]'  # Double backslash \\[...\\]
    ]
    
    for eq_pattern in equation_patterns:
        for match in re.finditer(eq_pattern, context, re.DOTALL):
            math_start = match.start()
            math_end = match.end()
            if math_start < relative_start and relative_end < math_end:
                return True, f"Inside equation: {match.group(0)[:30]}..."
    
    # 4. Check other LaTeX math environments
    math_environments = [
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
        r'\\begin\{align\*\}(.*?)\\end\{align\*\}',
        r'\\begin\{alignat\}(.*?)\\end\{alignat\}',
        r'\\begin\{alignat\*\}(.*?)\\end\{alignat\*\}',
        r'\\begin\{gather\}(.*?)\\end\{gather\}',
        r'\\begin\{gather\*\}(.*?)\\end\{gather\*\}',
        r'\\begin\{multline\}(.*?)\\end\{multline\}',
        r'\\begin\{multline\*\}(.*?)\\end\{multline\*\}',
        r'\\begin\{split\}(.*?)\\end\{split\}',
        r'\\begin\{cases\}(.*?)\\end\{cases\}',
        r'\\begin\{matrix\}(.*?)\\end\{matrix\}',
        r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}',
        r'\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}',
        r'\\begin\{vmatrix\}(.*?)\\end\{vmatrix\}',
        r'\\begin\{Vmatrix\}(.*?)\\end\{Vmatrix\}',
        # Support environments with double backslashes
        r'\\\\begin\{equation\}(.*?)\\\\end\{equation\}',
        r'\\\\begin\{equation\*\}(.*?)\\\\end\{equation\*\}',
        r'\\\\begin\{align\}(.*?)\\\\end\{align\}',
        r'\\\\begin\{align\*\}(.*?)\\\\end\{align\*\}',
        r'\\\\begin\{cases\}(.*?)\\\\end\{cases\}',
        r'\\\\begin\{matrix\}(.*?)\\\\end\{matrix\}',
        r'\\\\begin\{pmatrix\}(.*?)\\\\end\{pmatrix\}',
        r'\\\\begin\{bmatrix\}(.*?)\\\\end\{bmatrix\}',
        r'\\\\begin\{vmatrix\}(.*?)\\\\end\{vmatrix\}',
        r'\\\\begin\{Vmatrix\}(.*?)\\\\end\{Vmatrix\}',
    ]
    
    for env_pattern in math_environments:
        for match in re.finditer(env_pattern, context, re.DOTALL | re.IGNORECASE):
            math_start = match.start()
            math_end = match.end()
            if math_start < relative_start and relative_end < math_end:
                env_name = env_pattern.split(r'\{')[1].split(r'\}')[0] if r'\{' in env_pattern else 'unknown'
                return True, f"Inside LaTeX math environment: {env_name}"
    
    # 5. Check other common math markers
    math_mode_patterns = [
        r'\\begin\{math\}(.*?)\\end\{math\}',
        r'\\begin\{displaymath\}(.*?)\\end\{displaymath\}',
        r'\\ensuremath\{([^}]*)\}',
        # Support double backslashes
        r'\\\\begin\{math\}(.*?)\\\\end\{math\}',
        r'\\\\begin\{displaymath\}(.*?)\\\\end\{displaymath\}',
        r'\\\\ensuremath\{([^}]*)\}',
    ]
    
    for math_pattern in math_mode_patterns:
        for match in re.finditer(math_pattern, context, re.DOTALL | re.IGNORECASE):
            math_start = match.start()
            math_end = match.end()
            if math_start < relative_start and relative_end < math_end:
                return True, f"Inside math mode: {match.group(0)[:30]}..."
    
    return False, ""

def is_in_ambiguous_context(text, start_pos, end_pos):
    """Check if integer is in an ambiguous context (cannot be replaced at all)"""
    
    # Expand check range to see more context
    context_start = max(0, start_pos - 50)
    context_end = min(len(text), end_pos + 50)
    context = text[context_start:context_end]
    
    # Calculate relative position in context
    relative_start = start_pos - context_start
    relative_end = end_pos - context_start
    
    # 1. Enhanced subscript check
    subscript_patterns = [
        r'(\w+)_\{([^}]*)\}',      # x_{i + 1}, a_{i + number}
        r'(\w+)_\{?(\d+)\}?',      # x_1, x_{12}
        r'(\w+)_([a-zA-Z]\w*)',    # x_i, x_abc
        r'([a-zA-Z])\s*_\s*\{([^}]*)\}',  # x _ {i + 1}
        r'([a-zA-Z])\s*_\s*(\d+)', # x _ 1
    ]
    
    for sub_pattern in subscript_patterns:
        for match in re.finditer(sub_pattern, context):
            subscript_start = match.start(2)
            subscript_end = match.end(2)
            if subscript_start <= relative_start and relative_end <= subscript_end:
                return True, f"Part of subscript: {match.group(0)}"
    
    # 2. Enhanced superscript check
    superscript_patterns = [
        r'(\w+)\^\{([^}]*)\}',      # x^{n + 1}, a^{i + number}
        r'(\w+)\^\{?(\d+)\}?',      # x^1, x^{12}
        r'(\w+)\^([a-zA-Z]\w*)',    # x^n, x^abc
        r'([a-zA-Z])\s*\^\s*\{([^}]*)\}',  # x ^ {n + 1}
        r'([a-zA-Z])\s*\^\s*(\d+)', # x ^ 1
    ]
    
    for sup_pattern in superscript_patterns:
        for match in re.finditer(sup_pattern, context):
            superscript_start = match.start(2)
            superscript_end = match.end(2)
            if superscript_start <= relative_start and relative_end <= superscript_end:
                return True, f"Part of superscript: {match.group(0)}"
    
    # 3. Check compound expressions in parentheses
    brace_expr_patterns = [
        r'\{[^}]*\b(\d+)\b[^}]*\}',  # {i + number} 
        r'\([^)]*\b(\d+)\b[^)]*\)',  # (i + number)
    ]
    
    for brace_pattern in brace_expr_patterns:
        for match in re.finditer(brace_pattern, context):
            expr_start = match.start()
            expr_end = match.end()
            
            prefix_context = context[max(0, expr_start-10):expr_start]
            if re.search(r'[a-zA-Z\w]_$', prefix_context) or re.search(r'[a-zA-Z\w]\^$', prefix_context):
                digit_start = match.start(1)
                digit_end = match.end(1)
                if digit_start <= relative_start and relative_end <= digit_end:
                    return True, f"Part of subscript/superscript expression: {match.group(0)}"
    
    # 4. Check formatted large numbers
    formatted_number_patterns = [
        r'\$?\d{1,3}(?:[,{](?:[,}]\d{3})+|\d{3})+',  # $20{,}000 or 20,000
        r'\d+[,，]\d{3}(?:[,，]\d{3})*',  # 20,000 or 20，000
        r'\d+\{\s*,\s*\}\d+',  # 20{,}000
    ]
    
    for pattern in formatted_number_patterns:
        for match in re.finditer(pattern, context):
            if match.start() <= relative_start and relative_end <= match.end():
                return True, f"Part of formatted number: {match.group(0)}"
    
    # 5. Check serial numbers or labels
    label_patterns = [
        r'(?:Problem|Step|Figure|Table|Example|Question)\s+(\d+)',
        r'\((\d+)\)',  # (1), (2) style numbering
        r'(\d+)\.',    # 1., 2. style numbering
    ]
    
    for label_pattern in label_patterns:
        for match in re.finditer(label_pattern, context, re.IGNORECASE):
            label_start = match.start(1) 
            label_end = match.end(1)
            if label_start <= relative_start and relative_end <= label_end:
                return True, f"Part of label/index: {match.group(0)}"
    
    # 6. Check coordinates or point representations
    coordinate_patterns = [
        r'\(\s*-?\d+\s*,\s*-?\d+\s*\)',  # (3, 4), (-3, 4)
        r'\{\s*-?\d+\s*,\s*-?\d+\s*\}',  # {3, 4}, {-3, 4}
    ]
    
    for coord_pattern in coordinate_patterns:
        for match in re.finditer(coord_pattern, context):
            if match.start() <= relative_start and relative_end <= match.end():
                return True, f"Part of coordinate: {match.group(0)}"
    
    return False, ""

def extract_numbers_from_text(text):
    """Extract all potentially replaceable integers from text"""
    numbers_with_pos = []

    # First mark all position ranges of floating-point numbers
    float_ranges = set()

    # Match various floating-point formats
    float_patterns = [
        r'\d+\.\d+',  # 123.45
        r'\d+\.',     # 123.
        r'\.\d+',     # .45
    ]

    for pattern in float_patterns:
        for match in re.finditer(pattern, text):
            # Mark all positions covered by floating-point numbers
            for pos in range(match.start(), match.end()):
                float_ranges.add(pos)

    # Find all integers, excluding those in floating-point ranges
    integer_pattern = r'\d+'
    for match in re.finditer(integer_pattern, text):
        start_pos = match.start()
        end_pos = match.end()

        # Check if part of float
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

        # Check ambiguous contexts
        is_ambiguous, ambiguous_reason = is_in_ambiguous_context(text, start_pos, end_pos)
        if is_ambiguous:
            continue

        # Check complex expressions
        is_in_complex, complex_reason = is_in_complex_expression(text, start_pos, end_pos)

        # Check math environments
        is_in_math_env, math_env_reason = is_in_latex_math_environment(text, start_pos, end_pos)

        num = int(match.group())
        if num > 0:  # Keep positive integers only
            numbers_with_pos.append({
                'number': num,
                'start': start_pos,
                'end': end_pos,
                'text': match.group(),
                'context': text[max(0, start_pos-20):min(len(text), end_pos+20)],
                'is_independent': not is_in_complex,
                'complex_reason': complex_reason if is_in_complex else "",
                'is_in_math_env': is_in_math_env,
                'math_env_reason': math_env_reason if is_in_math_env else ""
            })

    return numbers_with_pos

class GPT_Client:
    def __init__(self, url, headers, model_name="gpt-4.1"):
        self.url = url
        self.headers = headers
        self.model_name = model_name
        self.rate_limit_lock = threading.Lock()
        self.api_call_count = 0
        self.api_fail_count = 0

    def _wait_for_rate_limit(self, min_interval=0.2):
        with self.rate_limit_lock:
            # Add random jitter
            jitter = random.uniform(0, min_interval * 0.5)
            time.sleep(min_interval + jitter)

    def call(self, user_prompt: str, max_retries=5):
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 16384,
        }
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                with self.rate_limit_lock:
                    self.api_call_count += 1
                
                response = requests.post(self.url, headers=self.headers, 
                                       data=json.dumps(data), timeout=120)
                
                if response.status_code == 429:
                    wait_time = min(60, 2 ** attempt + random.uniform(1, 3))
                    print(f"⏳ Rate limited, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                choices = result.get("choices", [])
                
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    reasoning_content = choices[0].get("message", {}).get("reasoning_content", "")
                    if reasoning_content:
                        reasoning_content = f""
                        result_content = reasoning_content + "\n\n" + content.lstrip()
                    else:
                        result_content = content
                    
                    return {"content": result_content, "success": True}
                    
            except Exception as ex:
                with self.rate_limit_lock:
                    self.api_fail_count += 1
                
                if attempt == max_retries - 1:
                    return {"content": None, "error": str(ex), "success": False}
                
                wait_time = min(30, 2 ** attempt + random.uniform(1, 2))
                print(f"⚠️ API error (attempt {attempt + 1}/{max_retries}): {str(ex)[:100]}")
                time.sleep(wait_time)
        
        return {"content": None, "error": "Max retries exceeded", "success": False}

# Global variables
processed_count = 0
successful_count = 0
error_count = 0
skipped_count = 0
lock = threading.Lock()

# Dynamic parameter control
class DynamicConfig:
    def __init__(self):
        self.batch_size = 4
        self.num_workers = 8
        self.min_interval = 0.2
        self.retry_delay = 5.0
        self.max_workers = 16
        self.min_workers = 1
        self.max_batch_size = 16
        self.min_batch_size = 1
    
    def adjust_on_success(self):
        """Increase concurrency on success"""
        if self.num_workers < self.max_workers:
            self.num_workers = min(self.max_workers, self.num_workers + 1)
        if self.batch_size < self.max_batch_size:
            self.batch_size = min(self.max_batch_size, self.batch_size + 1)
        if self.min_interval > 0.1:
            self.min_interval = max(0.1, self.min_interval * 0.9)
    
    def adjust_on_failure(self):
        """Reduce concurrency on failure"""
        if self.num_workers > self.min_workers:
            self.num_workers = max(self.min_workers, self.num_workers - 1)
        if self.batch_size > self.min_batch_size:
            self.batch_size = max(self.min_batch_size, self.batch_size - 1)
        self.min_interval = min(2.0, self.min_interval * 1.5)
        self.retry_delay = min(30.0, self.retry_delay * 1.2)

config = DynamicConfig()

def extract_key_variable_judgment(text: str) -> Dict[str, str]:
    """Extract key variable judgment results"""
    patterns = {
        'is_key_variable': [
            r'<is_key_variable>\s*(.*?)\s*</is_key_variable>',
            r'<is_key_variable>(.*?)</is_key_variable>'
        ],
        'confidence': [
            r'<confidence>\s*(.*?)\s*</confidence>',
            r'<confidence>(.*?)</confidence>'
        ],
        'reason': [
            r'<reason>\s*(.*?)\s*</reason>',
            r'<reason>(.*?)</reason>'
        ]
    }
    
    results = {}
    for field_name, pattern_list in patterns.items():
        found = False
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                content = match.group(1)
                if content is not None:
                    cleaned_content = content.strip()
                    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
                    results[field_name] = cleaned_content
                    found = True
                    break
            if found:
                break
        
        if not found:
            results[field_name] = "Not found"
    
    return results

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

def evaluate_all_numbers_as_key_variables(input_text, target, numbers_with_pos, gpt_client):
    """Evaluate all numbers as potential key variables"""
    evaluated_numbers = []
    
    for number_info in numbers_with_pos:
        selected_number = number_info['number']
        
        # Build judgment prompt
        user_prompt = f"""Problem: {input_text}

Target Answer: {target}

Selected Number: {selected_number}
Number Context: "{number_info['context']}"
Number Position: characters {number_info['start']}-{number_info['end']}
Is Independent: {number_info['is_independent']} (whether it can be independently replaced)
Is In Math Environment: {number_info['is_in_math_env']} (whether it's in LaTeX math environment)

Please analyze whether the selected number ({selected_number}) is a KEY VARIABLE for solving this problem.

A number is considered a KEY VARIABLE if:
1. The problem cannot be solved without this specific number
2. Changing this number would significantly change the final answer
3. This number is essential for the mathematical calculation or reasoning process

A number is NOT a key variable if:
1. It's just a label, index, or identifier (like "Problem 1", "Step 2")
2. It's a formatting element or example number
3. The problem can still be solved without this specific number
4. It's redundant information that doesn't affect the solution

Please provide your judgment with confidence level and reasoning:

<is_key_variable>YES/NO</is_key_variable>
<confidence>HIGH/MEDIUM/LOW</confidence>
<reason>Explain why this number is or isn't a key variable for solving the problem</reason>"""
        
        # Wait for rate limit
        gpt_client._wait_for_rate_limit(config.min_interval)
        
        api_result = gpt_client.call(user_prompt)
        
        if api_result and api_result.get("success"):
            api_output = api_result["content"]
            
            # Extract judgment results
            judgment_results = extract_key_variable_judgment(api_output)
            is_key_variable = judgment_results.get('is_key_variable', 'Not found')
            confidence = judgment_results.get('confidence', 'Not found')
            reason = judgment_results.get('reason', 'Not found')
            
            evaluated_numbers.append({
                "number": selected_number,
                "start_pos": number_info['start'],
                "end_pos": number_info['end'],
                "context": number_info['context'],
                "text": number_info['text'],
                "is_independent": number_info['is_independent'],
                "complex_reason": number_info.get('complex_reason', ''),
                "is_in_math_env": number_info['is_in_math_env'],
                "math_env_reason": number_info.get('math_env_reason', ''),
                "is_key_variable": is_key_variable,
                "confidence": confidence,
                "reason": reason,
                "api_output": api_output
            })
            
            independence = "INDEPENDENT" if number_info['is_independent'] else "COMPLEX"
            math_env = "MATH_ENV" if number_info['is_in_math_env'] else "NORMAL"
            print(f"[{get_current_time()}] 📊 Evaluated {selected_number}: {is_key_variable} ({confidence}, {independence}, {math_env})")
            
        else:
            error_msg = api_result.get("error", "API call failed") if api_result else "API call failed"
            print(f"[{get_current_time()}] ❌ API failed for number {selected_number} - {error_msg[:50]}")
            
            # Handle API failure
            evaluated_numbers.append({
                "number": selected_number,
                "start_pos": number_info['start'],
                "end_pos": number_info['end'], 
                "context": number_info['context'],
                "text": number_info['text'],
                "is_independent": number_info['is_independent'],
                "complex_reason": number_info.get('complex_reason', ''),
                "is_in_math_env": number_info['is_in_math_env'],
                "math_env_reason": number_info.get('math_env_reason', ''),
                "is_key_variable": "API failed",
                "confidence": "API failed",
                "reason": f"API call failed: {error_msg}",
                "api_output": f"API call failed: {error_msg}"
            })
    
    return evaluated_numbers

def find_best_key_variable(evaluated_numbers):
    """Select best key variable according to 4-level priority"""
    # Filter key variables
    key_variables = [num for num in evaluated_numbers if num['is_key_variable'].upper() == "YES"]
    
    if not key_variables:
        return None, "no_key_variable"
    
    # Priority 1: Independent and not in math environment
    priority1_vars = [num for num in key_variables 
                     if num['is_independent'] and not num['is_in_math_env']]
    
    if priority1_vars:
        priority1_vars.sort(key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x['confidence'].upper(), 0), reverse=True)
        return priority1_vars[0], "independent_non_math_env"
    
    # Priority 2: Independent and in math environment
    priority2_vars = [num for num in key_variables 
                     if num['is_independent'] and num['is_in_math_env']]
    
    if priority2_vars:
        priority2_vars.sort(key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x['confidence'].upper(), 0), reverse=True)
        return priority2_vars[0], "independent_in_math_env"
    
    # Priority 3: Complex and not in math environment
    priority3_vars = [num for num in key_variables 
                     if not num['is_independent'] and not num['is_in_math_env']]
    
    if priority3_vars:
        priority3_vars.sort(key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x['confidence'].upper(), 0), reverse=True)
        return priority3_vars[0], "complex_non_math_env"
    
    # Priority 4: Complex and in math environment
    priority4_vars = [num for num in key_variables 
                     if not num['is_independent'] and num['is_in_math_env']]
    
    if priority4_vars:
        priority4_vars.sort(key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x['confidence'].upper(), 0), reverse=True)
        return priority4_vars[0], "complex_in_math_env"
    
    # Should theoretically not reach here
    return None, "no_suitable_key_variable"

def find_key_variable_for_sample(sample_data, gpt_client: GPT_Client):
    """Find key variable for a single sample"""
    global processed_count, successful_count, error_count, skipped_count
    
    try:
        input_text = sample_data.get("input", "")
        instance_id = sample_data.get("instanceId", "")
        target = sample_data.get("target", [])
        pass_rate = sample_data.get("pass_rate", "")
        avg_output_tokens = sample_data.get("avg_output_tokens", "")
        
        if not input_text:
            with lock:
                processed_count += 1
                skipped_count += 1
            return None, "no_input"
        
        # Extract replaceable integers
        numbers_with_pos = extract_numbers_from_text(input_text)
        
        if not numbers_with_pos:
            with lock:
                processed_count += 1
                skipped_count += 1
            print(f"[{get_current_time()}] ⚠️ {instance_id}: No replaceable integers found, skipped")
            return None, "no_integers"
        
        print(f"[{get_current_time()}] 🔍 {instance_id}: Evaluating {len(numbers_with_pos)} numbers...")
        
        # Evaluate all numbers
        evaluated_numbers = evaluate_all_numbers_as_key_variables(input_text, target, numbers_with_pos, gpt_client)
        
        # Select best key variable
        best_choice, selection_strategy = find_best_key_variable(evaluated_numbers)
        
        if best_choice is None:
            with lock:
                processed_count += 1
                skipped_count += 1
            print(f"[{get_current_time()}] ⚠️ {instance_id}: {selection_strategy}, skipped")
            return None, selection_strategy
        
        result = {
            "instanceId": instance_id,
            "input": input_text,
            "target": target,
            "pass_rate": pass_rate,
            "avg_output_tokens": avg_output_tokens,
            "all_replaceable_numbers": numbers_with_pos,
            "selected_variable": {
                "number": best_choice['number'],
                "start_pos": best_choice['start_pos'],
                "end_pos": best_choice['end_pos'],
                "context": best_choice['context'],
                "text": best_choice['text'],
                "is_independent": best_choice['is_independent'],
                "complex_reason": best_choice['complex_reason'],
                "is_in_math_env": best_choice['is_in_math_env'],
                "math_env_reason": best_choice['math_env_reason']
            },
            "is_key_variable": best_choice['is_key_variable'],
            "confidence": best_choice['confidence'],
            "reason": best_choice['reason'],
            "selection_strategy": selection_strategy,
            "all_evaluated_numbers": evaluated_numbers,
            "api_output": best_choice['api_output'],
            "retry_count": sample_data.get('retry_count', 0)
        }
        
        with lock:
            processed_count += 1
            successful_count += 1
        
        independence = "INDEPENDENT" if best_choice['is_independent'] else "COMPLEX"
        math_env = "MATH_ENV" if best_choice['is_in_math_env'] else "NORMAL"
        strategy_map = {
            "independent_non_math_env": "P1",
            "independent_in_math_env": "P2", 
            "complex_non_math_env": "P3",
            "complex_in_math_env": "P4"
        }
        priority = strategy_map.get(selection_strategy, "P?")
        print(f"[{get_current_time()}] ✅ {instance_id} -> Selected key variable: {best_choice['number']} ({independence}, {math_env}, {priority})")
        return result, "success"
            
    except Exception as e:
        with lock:
            processed_count += 1
            error_count += 1
        
        print(f"[{get_current_time()}] ❌ {instance_id}: Processing error - {str(e)[:50]}")
        return None, "processing_error"

def batch_write_results(results, output_file):
    if not results:
        return
    
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for result in results:
                if result:
                    json_str = json.dumps(result, ensure_ascii=False, indent=2)
                    f.write(json_str + '\n\n')
        print(f"[{get_current_time()}] 💾 Saved {len(results)} results")
    except Exception as e:
        print(f"[{get_current_time()}] ❌ Write failed: {str(e)}")

def process_with_retry(data_list: List[dict], gpt_client: GPT_Client, output_file: str, max_retry_rounds: int = 10):
    """Process data with retry support"""
    global processed_count, successful_count, error_count, skipped_count, config
    
    retry_queue = data_list.copy()
    retry_round = 0
    
    while retry_queue and retry_round < max_retry_rounds:
        retry_round += 1
        current_queue = retry_queue.copy()
        retry_queue = []
        
        print(f"\n🔄 Retry Round {retry_round}: Processing {len(current_queue)} samples")
        print(f"⚙️ Config: workers={config.num_workers}, batch_size={config.batch_size}, interval={config.min_interval:.2f}s")
        
        batch_results = []
        batch_success_count = 0
        batch_fail_count = 0
        batch_skip_count = 0
        
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            # Add retry count
            for sample in current_queue:
                sample['retry_count'] = sample.get('retry_count', 0) + 1
            
            future_to_data = {executor.submit(find_key_variable_for_sample, sample, gpt_client): sample 
                            for sample in current_queue}
            
            for future in as_completed(future_to_data):
                try:
                    result, status = future.result()
                    original_sample = future_to_data[future]
                    
                    if status == "success":
                        batch_results.append(result)
                        batch_success_count += 1
                        
                        # Write in batches
                        if len(batch_results) >= config.batch_size:
                            batch_write_results(batch_results, output_file)
                            batch_results = []
                            
                    elif status in ["no_input", "no_integers", "no_key_variable", "no_suitable_key_variable"]:
                        batch_skip_count += 1
                        
                    elif status in ["processing_error"]:
                        batch_fail_count += 1
                        # Retry up to 3 times
                        if original_sample['retry_count'] < 3:
                            retry_queue.append(original_sample)
                            
                except Exception as e:
                    print(f"❌ Task execution error: {str(e)}")
        
        # Write remaining results
        if batch_results:
            batch_write_results(batch_results, output_file)
        
        print(f"📊 Round {retry_round} Summary: Success={batch_success_count}, Failed={batch_fail_count}, Skipped={batch_skip_count}, Remaining={len(retry_queue)}")
        print(f"🔧 API Stats: Total calls={gpt_client.api_call_count}, Failures={gpt_client.api_fail_count}")
        
        # Adjust parameters dynamically
        if batch_fail_count > batch_success_count:
            config.adjust_on_failure()
            print(f"⬇️ Reducing concurrency due to high failure rate")
        elif batch_fail_count == 0 and batch_success_count > 0:
            config.adjust_on_success()
            print(f"⬆️ Increasing concurrency due to high success rate")
        
        # Wait before next retry
        if retry_queue:
            wait_time = min(60, config.retry_delay * (retry_round ** 0.5))
            print(f"⏳ Waiting {wait_time:.1f}s before next retry round...")
            time.sleep(wait_time)
    
    return len(data_list) - len(retry_queue)

def find_key_variables(input_file: str, output_file: str, gpt_client: GPT_Client):
    global processed_count, successful_count, error_count, skipped_count, config
    processed_count = successful_count = error_count = skipped_count = 0
    
    print(f"🚀 Finding key variables with 4-level priority strategy: {os.path.basename(input_file)}")
    print(f"🤖 Model: {gpt_client.model_name}")
    print(f"🔄 4-Level Priority Strategy (Only key variables are eligible):")
    print(f"   1️⃣ Priority: Independent key variable NOT in math environment")
    print(f"   2️⃣ Priority: Independent key variable IN math environment")
    print(f"   3️⃣ Priority: Complex key variable NOT in math environment")
    print(f"   4️⃣ Priority: Complex key variable IN math environment")
    print(f"   ❌ Skip: Non-key variables")
    print(f"   🚫 Exclude: Subscripts (x_1), superscripts (x^1), labels, coordinates")
    
    try:
        # Read filtered data
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # Get examples array
        examples = data.get('examples', [])
        print(f"📊 Total samples: {len(examples)}")
        
        if not examples:
            print("❌ No samples to analyze")
            return
        
        if os.path.exists(output_file):
            os.remove(output_file)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Reset configuration
        config = DynamicConfig()
        
        # Process with retry mechanism
        successful_samples = process_with_retry(examples, gpt_client, output_file)
        
        print(f"\n✅ Processing Complete!")
        print(f"📈 Final Stats:")
        print(f"   - Total processed: {processed_count}")
        print(f"   - Found key variables: {successful_count}")
        print(f"   - Skipped: {skipped_count}")
        print(f"   - Failed: {error_count}")
        print(f"   - Key variable success rate: {successful_count}/{len(examples)} ({successful_count/len(examples)*100:.1f}%)")
        print(f"   - Total API calls: {gpt_client.api_call_count}")
        print(f"   - API failure rate: {gpt_client.api_fail_count}/{max(1, gpt_client.api_call_count)} ({gpt_client.api_fail_count/max(1, gpt_client.api_call_count)*100:.1f}%)")
        print(f"💾 Output: {output_file}")
        
        # Generate statistics
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                p1_strategy = content.count('"selection_strategy": "independent_non_math_env"')
                p2_strategy = content.count('"selection_strategy": "independent_in_math_env"')
                p3_strategy = content.count('"selection_strategy": "complex_non_math_env"')
                p4_strategy = content.count('"selection_strategy": "complex_in_math_env"')
                
                high_conf = content.count('"confidence": "HIGH"')
                medium_conf = content.count('"confidence": "MEDIUM"')
                low_conf = content.count('"confidence": "LOW"')
                
                in_math_env_count = content.count('"is_in_math_env": true')
                not_in_math_env_count = content.count('"is_in_math_env": false')
                
                independent_count = content.count('"is_independent": true')
                complex_count = content.count('"is_independent": false')
                
                total_strategies = p1_strategy + p2_strategy + p3_strategy + p4_strategy
                total_conf = high_conf + medium_conf + low_conf
                total_math_env = in_math_env_count + not_in_math_env_count
                total_independence = independent_count + complex_count
                
                if total_strategies > 0:
                    print(f"📊 4-Level Priority Distribution:")
                    print(f"   - P1: {p1_strategy} ({p1_strategy/total_strategies*100:.1f}%)")
                    print(f"   - P2: {p2_strategy} ({p2_strategy/total_strategies*100:.1f}%)")
                    print(f"   - P3: {p3_strategy} ({p3_strategy/total_strategies*100:.1f}%)")
                    print(f"   - P4: {p4_strategy} ({p4_strategy/total_strategies*100:.1f}%)")
                
                if total_conf > 0:
                    print(f"📊 Confidence Distribution:")
                    print(f"   - HIGH: {high_conf} ({high_conf/total_conf*100:.1f}%)")
                    print(f"   - MEDIUM: {medium_conf} ({medium_conf/total_conf*100:.1f}%)")
                    print(f"   - LOW: {low_conf} ({low_conf/total_conf*100:.1f}%)")
                
                if total_math_env > 0:
                    print(f"📊 Math Environment Distribution:")
                    print(f"   - In math environment: {in_math_env_count} ({in_math_env_count/total_math_env*100:.1f}%)")
                    print(f"   - Not in math environment: {not_in_math_env_count} ({not_in_math_env_count/total_math_env*100:.1f}%)")
                
                if total_independence > 0:
                    print(f"📊 Independence Distribution:")
                    print(f"   - Independent: {independent_count} ({independent_count/total_independence*100:.1f}%)")
                    print(f"   - Complex: {complex_count} ({complex_count/total_independence*100:.1f}%)")
                    
        except:
            pass
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        traceback.print_exc()

def main():
    # API configuration - users must replace with their own
    url = "https://api.openai.com/v1/chat/completions"  # OpenAI-compatible API
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"  # Replace with your API key
    }
    
    model_name = "gpt-4.1"  # Default model
    gpt_client = GPT_Client(url=url, headers=headers, model_name=model_name)
    
    # File path configuration
    input_file = "./data_contruction/data/filtered_samples.jsonl"  # From previous filtering step
    output_file = "./data_contruction/data/key_variables_results.jsonl"  # Output file

    # Check input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Start analysis
    find_key_variables(input_file, output_file, gpt_client)

if __name__ == "__main__":
    main()
    