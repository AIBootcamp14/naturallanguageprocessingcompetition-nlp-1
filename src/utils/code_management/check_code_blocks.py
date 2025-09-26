#!/usr/bin/env python3
import os
import re

def check_code_blocks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all code block markers
    code_blocks = re.findall(r'^```.*?$', content, re.MULTILINE)
    
    # Count opening and closing markers
    opening = len([block for block in code_blocks if not block.strip() == '```'])
    closing = len([block for block in code_blocks if block.strip() == '```'])
    
    return len(code_blocks), opening, closing

# Check all markdown files in docs folder
docs_files = []
for root, dirs, files in os.walk('docs'):
    for file in files:
        if file.endswith('.md'):
            docs_files.append(os.path.join(root, file))

print('파일별 코드 블록 체크:')
print('=' * 80)
problems_found = []

for file_path in sorted(docs_files):
    total, opening, closing = check_code_blocks(file_path)
    status = '✅' if opening == closing else '❌'
    print(f'{status} {file_path}')
    print(f'   전체: {total}, 열기: {opening}, 닫기: {closing}')
    if opening != closing:
        print(f'   ⚠️ 불일치! 차이: {opening - closing}')
        problems_found.append(file_path)
    print()

if problems_found:
    print(f'\n🚨 문제가 있는 파일들: {len(problems_found)}개')
    for file in problems_found:
        print(f'   - {file}')
else:
    print(f'\n✅ 모든 파일의 코드 블록이 올바르게 닫혀있습니다!')
