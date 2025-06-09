#!/usr/bin/env python3
"""
Simple script to check for potentially unused imports in Python files.
"""

import os
import re
import sys
from pathlib import Path

def check_file_imports(file_path):
    """Check a Python file for potentially unused imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    # Find import statements
    import_pattern = r'^(?:from\s+[\w.]+\s+)?import\s+(.*?)(?:\s+as\s+\w+)?$'
    imports = []
    
    for line_num, line in enumerate(content.split('\n'), 1):
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            imports.append((line_num, line))
    
    unused = []
    
    for line_num, import_line in imports:
        # Extract imported names
        if import_line.startswith('from '):
            # from module import name1, name2
            match = re.match(r'from\s+[\w.]+\s+import\s+(.*)', import_line)
            if match:
                imported_names = [name.strip().split(' as ')[0].strip() for name in match.group(1).split(',')]
        else:
            # import module or import module as alias
            match = re.match(r'import\s+(.*)', import_line)
            if match:
                imported_names = []
                for item in match.group(1).split(','):
                    item = item.strip()
                    if ' as ' in item:
                        # import module as alias - use alias
                        imported_names.append(item.split(' as ')[1].strip())
                    else:
                        # import module - use module name
                        if '.' in item:
                            imported_names.append(item.split('.')[0])
                        else:
                            imported_names.append(item)
        
        # Check if any of the imported names are used
        for name in imported_names:
            if name == '*':
                continue  # Skip wildcard imports
                
            # Look for usage in the content (excluding the import line itself)
            content_without_imports = '\n'.join([l for l in content.split('\n') if not (l.strip().startswith('import ') or l.strip().startswith('from '))])
            
            # Check for usage patterns
            usage_patterns = [
                rf'\b{re.escape(name)}\.',  # module.method or module.attribute
                rf'\b{re.escape(name)}\(',  # function call
                rf'\b{re.escape(name)}\[',  # indexing
                rf'isinstance\([^,]+,\s*{re.escape(name)}\)',  # isinstance check
                rf'raise\s+{re.escape(name)}',  # raise Exception
                rf'except\s+.*{re.escape(name)}',  # except Exception
                rf':\s*{re.escape(name)}$',  # type hints at end of line
                rf':\s*{re.escape(name)}\s*\)',  # type hints in function params
                rf'->\s*{re.escape(name)}',  # return type hints
                rf'Union\[.*{re.escape(name)}',  # Union type hints
                rf'Optional\[.*{re.escape(name)}',  # Optional type hints
                rf'List\[.*{re.escape(name)}',  # List type hints
                rf'Dict\[.*{re.escape(name)}',  # Dict type hints
            ]
            
            used = False
            for pattern in usage_patterns:
                if re.search(pattern, content_without_imports, re.MULTILINE):
                    used = True
                    break
            
            if not used:
                unused.append((line_num, import_line, name))
    
    return unused

def main():
    """Main function to check imports in bookmark_processor directory."""
    base_dir = Path("/mnt/c/Users/Troy Davis/OneDrive/Projects/Code/Python/bookmark-validator/bookmark_processor")
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Find all Python files
    python_files = list(base_dir.rglob("*.py"))
    
    all_unused = []
    
    for py_file in python_files:
        unused = check_file_imports(py_file)
        if unused:
            all_unused.extend([(py_file, line_num, import_line, name) for line_num, import_line, name in unused])
    
    if all_unused:
        print("Potentially unused imports found:")
        print("=" * 50)
        
        current_file = None
        for file_path, line_num, import_line, name in all_unused:
            if file_path != current_file:
                current_file = file_path
                print(f"\n{file_path}:")
            print(f"  Line {line_num}: {import_line} (unused: {name})")
    else:
        print("No unused imports found!")

if __name__ == "__main__":
    main()