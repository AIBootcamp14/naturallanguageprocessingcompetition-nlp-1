#!/usr/bin/env python3
import os
import re

def fix_code_blocks_advanced(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all code block markers with their positions
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a code block
        if re.match(r'^```\w*', line.strip()):
            fixed_lines.append(line)
            i += 1
            
            # Look for closing code block
            found_closing = False
            while i < len(lines):
                current_line = lines[i]
                
                # Found proper closing
                if current_line.strip() == '```':
                    fixed_lines.append(current_line)
                    found_closing = True
                    break
                
                # Found patterns that suggest we need to close the code block
                elif (re.match(r'^#+\s', current_line) or  # Headers
                      re.match(r'^---+', current_line) or  # Horizontal rules  
                      re.match(r'^\*\*.*\*\*:?', current_line) or  # Bold text
                      re.match(r'^###?\s.*:$', current_line) or  # Section headers
                      re.match(r'^##?\s', current_line)):  # Any header
                    
                    # Insert closing before this line
                    fixed_lines.append('```')
                    fixed_lines.append(current_line)
                    found_closing = True
                    break
                else:
                    fixed_lines.append(current_line)
                    i += 1
            
            # If no closing found, add one
            if not found_closing:
                fixed_lines.append('```')
        else:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)

# Get all problematic files
problematic_files = []
for root, dirs, files in os.walk('docs'):
    for file in files:
        if file.endswith('.md'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            code_blocks = re.findall(r'^```.*?$', content, re.MULTILINE)
            opening = len([block for block in code_blocks if not block.strip() == '```'])
            closing = len([block for block in code_blocks if block.strip() == '```'])
            
            if opening != closing:
                problematic_files.append(file_path)

print(f"Found {len(problematic_files)} problematic files. Fixing...")

for file_path in problematic_files:
    print(f"Fixing {file_path}...")
    try:
        fixed_content = fix_code_blocks_advanced(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"✅ Fixed {file_path}")
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")

print("\n🔍 Re-checking all files...")
# Re-check
remaining_problems = []
for root, dirs, files in os.walk('docs'):
    for file in files:
        if file.endswith('.md'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            code_blocks = re.findall(r'^```.*?$', content, re.MULTILINE)
            opening = len([block for block in code_blocks if not block.strip() == '```'])
            closing = len([block for block in code_blocks if block.strip() == '```'])
            
            if opening != closing:
                remaining_problems.append((file_path, opening, closing))

if remaining_problems:
    print(f"\n🚨 Still {len(remaining_problems)} files with problems:")
    for file_path, opening, closing in remaining_problems:
        print(f"   {file_path}: {opening} opening, {closing} closing (diff: {opening-closing})")
else:
    print(f"\n✅ All files fixed! No more code block problems.")
