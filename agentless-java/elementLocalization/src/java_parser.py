# src/java_parser.py
import os
import javalang
from typing import Dict, List, Any, Optional

def load_java_file_content(repo_path, file_path):
    """
    Load Java file content from repository.
    """
    full_path = os.path.join(repo_path, file_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {full_path}: {e}")
        return None

def extract_skeleton(repo_path, file_path):
    """
    Extract the skeleton format from a Java file.
    """
    content = load_java_file_content(repo_path, file_path)
    if not content:
        return {"file_path": file_path, "skeleton": "", "error": "Failed to load file"}
    
    try:
        ast = javalang.parse.parse(content)
        
        package = ast.package.name if ast.package else "default"
        imports = [imp.path for imp in ast.imports]
        
        classes = []
        for path, node in ast.filter(javalang.tree.ClassDeclaration):
            class_info = {
                "name": node.name,
                "modifiers": node.modifiers,
                "extends": node.extends.name if node.extends else None,
                "implements": [i.name for i in node.implements] if node.implements else [],
                "fields": [],
                "methods": [],
                "comments": extract_javadoc(node)
            }
            
            # Extract fields
            for field_decl in node.fields:
                for field_var in field_decl.declarators:
                    class_info["fields"].append({
                        "name": field_var.name,
                        "type": field_decl.type.name,
                        "modifiers": field_decl.modifiers
                    })
            
            # Extract methods
            for method_decl in node.methods:
                class_info["methods"].append({
                    "name": method_decl.name,
                    "modifiers": method_decl.modifiers,
                    "return_type": method_decl.return_type.name if method_decl.return_type else "void",
                    "parameters": [{"type": p.type.name, "name": p.name} for p in method_decl.parameters],
                    "comments": extract_javadoc(method_decl)
                })
            
            classes.append(class_info)
        
        # Also handle interfaces, enums, etc.
        interfaces = []
        for path, node in ast.filter(javalang.tree.InterfaceDeclaration):
            interface_info = {
                "name": node.name,
                "modifiers": node.modifiers,
                "extends": [i.name for i in node.extends] if node.extends else [],
                "methods": [],
                "comments": extract_javadoc(node)
            }
            
            # Extract methods
            for method_decl in node.methods:
                interface_info["methods"].append({
                    "name": method_decl.name,
                    "modifiers": method_decl.modifiers,
                    "return_type": method_decl.return_type.name if method_decl.return_type else "void",
                    "parameters": [{"type": p.type.name, "name": p.name} for p in method_decl.parameters],
                    "comments": extract_javadoc(method_decl)
                })
            
            interfaces.append(interface_info)
        
        return {
            "file_path": file_path,
            "package": package,
            "imports": imports,
            "classes": classes,
            "interfaces": interfaces,
            "error": None
        }
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return extract_skeleton_fallback(file_path, content)


def extract_javadoc(node):
    """Extract javadoc comments from a node if available"""
    if hasattr(node, 'documentation'):
        return node.documentation
    return None
def format_skeleton(skeleton_data):
    """
    Format the skeleton data into a readable string format with enhanced context.
    Provides better structure and additional information to help the LLM identify relevant elements.
    """
    if skeleton_data.get("error") and not skeleton_data.get("package"):
        # Try to recover some information even with errors
        if "skeleton" in skeleton_data and skeleton_data["skeleton"]:
            return skeleton_data["skeleton"]
        return f"// File: {skeleton_data['file_path']}\n// Error: {skeleton_data['error']}"
    
    output = []
    
    # Add file path with a more prominent header
    output.append(f"// ===============================================")
    output.append(f"// FILE: {skeleton_data['file_path']}")
    output.append(f"// ===============================================")
    output.append("")
    
    # Add package
    if skeleton_data.get('package'):
        output.append(f"package {skeleton_data['package']};")
        output.append("")
    
    # Add imports (grouped by domain for better readability)
    if skeleton_data.get('imports'):
        java_imports = []
        javax_imports = []
        org_imports = []
        other_imports = []
        
        for imp in skeleton_data.get('imports', []):
            if imp.startswith('java.'):
                java_imports.append(imp)
            elif imp.startswith('javax.'):
                javax_imports.append(imp)
            elif imp.startswith('org.'):
                org_imports.append(imp)
            else:
                other_imports.append(imp)
        
        for imp_list in [java_imports, javax_imports, org_imports, other_imports]:
            if imp_list:
                for imp in sorted(imp_list):
                    output.append(f"import {imp};")
                output.append("")
    
    # Add classes
    for cls in skeleton_data.get('classes', []):
        # Add class comments
        if cls.get('comments'):
            output.append(f"/**")
            for line in cls['comments'].split('\n'):
                output.append(f" * {line}")
            output.append(f" */")
        
        # Add class declaration
        modifiers = ' '.join(cls.get('modifiers', []))
        extends = f" extends {cls['extends']}" if cls.get('extends') else ""
        implements = f" implements {', '.join(cls['implements'])}" if cls.get('implements') and len(cls['implements']) > 0 else ""
        
        output.append(f"{modifiers} class {cls['name']}{extends}{implements} {{")
        
        # Add fields with better formatting
        if cls.get('fields'):
            output.append("    // Fields")
            for field in cls.get('fields', []):
                modifiers = ' '.join(field.get('modifiers', []))
                output.append(f"    {modifiers} {field['type']} {field['name']};")
            output.append("")
        
        # Add constructors (if separating them from regular methods)
        constructors = [m for m in cls.get('methods', []) if m['name'] == cls['name']]
        if constructors:
            output.append("    // Constructors")
            for constructor in constructors:
                # Add constructor comments
                if constructor.get('comments'):
                    output.append(f"    /**")
                    for line in constructor['comments'].split('\n'):
                        output.append(f"     * {line}")
                    output.append(f"     */")
                
                modifiers = ' '.join(constructor.get('modifiers', []))
                params = ", ".join([f"{p['type']} {p['name']}" for p in constructor.get('parameters', [])])
                output.append(f"    {modifiers} {cls['name']}({params}) {{...}}")
                output.append("")
        
        # Add methods (excluding constructors)
        methods = [m for m in cls.get('methods', []) if m['name'] != cls['name']]
        if methods:
            output.append("    // Methods")
            for method in methods:
                # Add method comments
                if method.get('comments'):
                    output.append(f"    /**")
                    for line in method['comments'].split('\n'):
                        output.append(f"     * {line}")
                    output.append(f"     */")
                
                modifiers = ' '.join(method.get('modifiers', []))
                return_type = method.get('return_type', 'void')
                params = ", ".join([f"{p['type']} {p['name']}" for p in method.get('parameters', [])])
                output.append(f"    {modifiers} {return_type} {method['name']}({params}) {{...}}")
                output.append("")
        
        output.append("}")
        output.append("")
    
    # Add interfaces
    for interface in skeleton_data.get('interfaces', []):
        # Add interface comments
        if interface.get('comments'):
            output.append(f"/**")
            for line in interface['comments'].split('\n'):
                output.append(f" * {line}")
            output.append(f" */")
        
        # Add interface declaration
        modifiers = ' '.join(interface.get('modifiers', []))
        extends = f" extends {', '.join(interface['extends'])}" if interface.get('extends') and len(interface['extends']) > 0 else ""
        
        output.append(f"{modifiers} interface {interface['name']}{extends} {{")
        
        # Add methods
        for method in interface.get('methods', []):
            # Add method comments
            if method.get('comments'):
                output.append(f"    /**")
                for line in method['comments'].split('\n'):
                    output.append(f"     * {line}")
                output.append(f"     */")
            
            modifiers = ' '.join(method.get('modifiers', []))
            return_type = method.get('return_type', 'void')
            params = ", ".join([f"{p['type']} {p['name']}" for p in method.get('parameters', [])])
            output.append(f"    {modifiers} {return_type} {method['name']}({params});")
            output.append("")
        
        output.append("}")
        output.append("")
    
    # Add enums if present
    for enum in skeleton_data.get('enums', []):
        modifiers = ' '.join(enum.get('modifiers', []))
        output.append(f"{modifiers} enum {enum['name']} {{")
        
        # Add enum constants
        if enum.get('constants'):
            constants_str = ', '.join(enum['constants'])
            output.append(f"    {constants_str}")
            
        # Add enum methods if any
        for method in enum.get('methods', []):
            modifiers = ' '.join(method.get('modifiers', []))
            return_type = method.get('return_type', 'void')
            params = ", ".join([f"{p['type']} {p['name']}" for p in method.get('parameters', [])])
            output.append(f"    {modifiers} {return_type} {method['name']}({params}) {{...}}")
            
        output.append("}")
        output.append("")
    
    # Add a summary at the end about what this file contains
    output.append(f"// =============== SUMMARY: {os.path.basename(skeleton_data['file_path'])} ===============")
    
    class_names = [cls['name'] for cls in skeleton_data.get('classes', [])]
    interface_names = [intf['name'] for intf in skeleton_data.get('interfaces', [])]
    enum_names = [enum['name'] for enum in skeleton_data.get('enums', [])] if 'enums' in skeleton_data else []
    
    if class_names:
        output.append(f"// Classes: {', '.join(class_names)}")
        # Count total methods across all classes
        total_methods = sum(len(cls.get('methods', [])) for cls in skeleton_data.get('classes', []))
        output.append(f"// Total methods: {total_methods}")
    
    if interface_names:
        output.append(f"// Interfaces: {', '.join(interface_names)}")
    
    if enum_names:
        output.append(f"// Enums: {', '.join(enum_names)}")
    
    # Identify potential main classes
    main_classes = []
    for cls in skeleton_data.get('classes', []):
        for method in cls.get('methods', []):
            if method['name'] == 'main' and 'static' in method.get('modifiers', []):
                params = method.get('parameters', [])
                if any(p.get('type') == 'String[]' for p in params):
                    main_classes.append(cls['name'])
    
    if main_classes:
        output.append(f"// Contains main method(s) in: {', '.join(main_classes)}")
    
    output.append(f"// ===============================================")
    
    return "\n".join(output)

def extract_skeleton_fallback(file_path, content):
    """Simplified parsing using regex when full parsing fails"""
    import re
    
    # Get package
    package_match = re.search(r'package\s+([^;]+);', content)
    package = package_match.group(1) if package_match else "default"
    
    # Get imports
    imports = re.findall(r'import\s+([^;]+);', content)
    
    # Get class/interface declarations
    class_pattern = r'(public|private|protected)?\s*(abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{'
    interface_pattern = r'(public|private|protected)?\s*interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*\{'
    
    classes = []
    for match in re.finditer(class_pattern, content):
        modifiers = [mod for mod in [match.group(1), match.group(2)] if mod]
        classes.append({
            "name": match.group(3),
            "modifiers": modifiers,
            "extends": match.group(4),
            "implements": [i.strip() for i in match.group(5).split(',')] if match.group(5) else [],
            "fields": [],
            "methods": []
        })
    
    # Add method extraction with regex
    method_pattern = r'(public|private|protected)?\s*(static|abstract|final)?\s*(?:<[^>]+>\s*)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)'
    
    # For each class, try to find methods
    for cls in classes:
        cls_start = content.find(f"class {cls['name']}")
        if cls_start == -1:
            continue
            
        # Find class end (imperfect but better than nothing)
        brace_count = 0
        cls_end = cls_start
        for i in range(cls_start, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    cls_end = i
                    break
        
        class_content = content[cls_start:cls_end]
        
        # Extract methods within class content
        for m_match in re.finditer(method_pattern, class_content):
            modifiers = [mod for mod in [m_match.group(1), m_match.group(2)] if mod]
            return_type = m_match.group(3)
            method_name = m_match.group(4)
            params_str = m_match.group(5)
            
            # Parse parameters
            params = []
            if params_str.strip():
                param_entries = params_str.split(',')
                for p in param_entries:
                    p = p.strip()
                    if not p:
                        continue
                    parts = p.split()
                    if len(parts) >= 2:
                        params.append({
                            "type": ' '.join(parts[:-1]),
                            "name": parts[-1]
                        })
            
            cls["methods"].append({
                "name": method_name,
                "modifiers": modifiers,
                "return_type": return_type,
                "parameters": params,
            })
    
    return {
        "file_path": file_path,
        "package": package,
        "imports": imports,
        "classes": classes,
        "interfaces": [],
        "error": "Used fallback parser"
    }