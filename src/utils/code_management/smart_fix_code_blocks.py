#!/usr/bin/env python3
import os
import re

def smart_fix_code_blocks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by lines for easier processing
    lines = content.split('\n')
    fixed_lines = []
    in_code_block = False
    code_block_start_line = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a code block
        if re.match(r'^```\w*', line.strip()) and not in_code_block:
            in_code_block = True
            code_block_start_line = i
            fixed_lines.append(line)
        
        # Check if this line ends a code block
        elif line.strip() == '```' and in_code_block:
            in_code_block = False
            fixed_lines.append(line)
        
        # Regular line in code block
        elif in_code_block:
            # Check if we hit a section header that suggests the code block should end
            if re.match(r'^#+\s', line) or re.match(r'^---+', line):
                # Close the code block before the header
                fixed_lines.append('```')
                in_code_block = False
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Regular line outside code block
        else:
            # Check for orphaned closing ```
            if line.strip() == '```':
                # Skip orphaned closing tags
                pass
            else:
                fixed_lines.append(line)
        
        i += 1
    
    # If still in code block at end of file, close it
    if in_code_block:
        fixed_lines.append('```')
    
    return '\n'.join(fixed_lines)

# Fix HIGH priority files
high_priority_files = [
    'docs/pipelines/전체_파이프라인_가이드.md',
    'docs/optimization/통합_실행_가이드.md',
]

for file_path in high_priority_files:
    if os.path.exists(file_path):
        print(f"🔧 Fixing {file_path}...")
        try:
            original_content = ""
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            fixed_content = smart_fix_code_blocks(file_path)
            
            # Only write if content changed
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"✅ Fixed {file_path}")
            else:
                print(f"📝 No changes needed for {file_path}")
                
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    else:
        print(f"⚠️ File not found: {file_path}")
