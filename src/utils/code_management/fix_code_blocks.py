#!/usr/bin/env python3
import os
import re

def fix_code_blocks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by lines for easier processing
    lines = content.split('\n')
    fixed_lines = []
    in_code_block = False
    code_block_language = ""
    
    for i, line in enumerate(lines):
        # Check if this line starts a code block
        if line.strip().startswith('```') and not in_code_block:
            in_code_block = True
            code_block_language = line.strip()[3:] if len(line.strip()) > 3 else ""
            fixed_lines.append(line)
        
        # Check if this line ends a code block
        elif line.strip() == '```' and in_code_block:
            in_code_block = False
            code_block_language = ""
            fixed_lines.append(line)
        
        # Regular line in code block
        elif in_code_block:
            fixed_lines.append(line)
        
        # Regular line outside code block
        else:
            fixed_lines.append(line)
            
            # Check if we need to close a code block before certain patterns
            if in_code_block:
                # Look ahead for patterns that suggest code block should end
                next_patterns = [
                    r'^#+\s',  # Headers
                    r'^---',   # Horizontal rules
                    r'^###\s.*:$',  # Section headers
                    r'^\*\*.*\*\*:',  # Bold headers
                ]
                
                for pattern in next_patterns:
                    if re.match(pattern, line):
                        # Insert closing code block
                        fixed_lines.insert(-1, '```')
                        in_code_block = False
                        code_block_language = ""
                        break
    
    # If still in code block at end of file, close it
    if in_code_block:
        fixed_lines.append('```')
    
    return '\n'.join(fixed_lines)

# Fix the most problematic files first
problem_files = [
    'docs/experiments/Swin_추론_파이프라인_결과_보고서.md',
    'docs/optimization/Optuna_하이퍼파라미터_최적화_가이드.md',
    'docs/optimization/Temperature_Scaling_캘리브레이션_가이드.md',
    'docs/experiments/모델_성능_비교_분석_보고서.md',
    'docs/experiments/전체_파이프라인_실행_결과_보고서.md',
]

for file_path in problem_files:
    if os.path.exists(file_path):
        print(f"Fixing {file_path}...")
        try:
            fixed_content = fix_code_blocks(file_path)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✅ Fixed {file_path}")
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    else:
        print(f"⚠️ File not found: {file_path}")
