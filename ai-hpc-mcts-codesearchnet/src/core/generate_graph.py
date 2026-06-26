import ast
import json
from typing import List, Dict, Any, Set, Optional
import re
from langchain_neo4j import Neo4jGraph
from tree_sitter_languages import get_parser
import os
import tree_sitter

# All import utility functions are now integrated below

# Standard library module names (moved from find_imports_treesitter.py)
try:
    with open("/Users/pratik.shah1/work/GraphRAG/buil_in_packages.txt", "r") as f:
        stdlib_modules = set(line.strip() for line in f if line.strip())
except FileNotFoundError:
    # Fallback list if file doesn't exist
    stdlib_modules = {
        "abc",
        "argparse",
        "array",
        "asyncio",
        "base64",
        "binascii",
        "bisect",
        "builtins",
        "cmath",
        "collections",
        "concurrent",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "dis",
        "enum",
        "functools",
        "gc",
        "getopt",
        "getpass",
        "gettext",
        "glob",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "keyword",
        "logging",
        "math",
        "multiprocessing",
        "numbers",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "plistlib",
        "queue",
        "random",
        "re",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "ssl",
        "statistics",
        "string",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "traceback",
        "types",
        "typing",
        "unicodedata",
        "urllib",
        "uuid",
        "warnings",
        "weakref",
        "xml",
        "zipfile",
    }


def load_requirements(requirements_path="requirements.txt"):
    """Load package names from requirements.txt"""
    if not os.path.exists(requirements_path):
        print(f"Warning: requirements file '{requirements_path}' not found.")
        return set()

    with open(requirements_path, "r") as f:
        lines = f.readlines()

    pkgs = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Remove version specifiers (e.g., 'numpy>=1.21')
        pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
        # Some packages use dashes in pip but are imported with underscores
        pkgs.add(pkg.replace("-", "_"))

    return pkgs


def is_installed_package(module_name, requirements_packages):
    """Check if a module is an installed package"""
    top_level = module_name.split(".")[0]
    return top_level in requirements_packages


def categorize_imports(imports, requirements_packages):
    """Categorize imports into stdlib, installed packages, and local imports"""
    stdlib = []
    installed = []
    local = []

    for imp in imports:
        top_level = imp.split(".")[0]
        if top_level in stdlib_modules:
            stdlib.append(imp)
        elif is_installed_package(top_level, requirements_packages):
            installed.append(imp)
        else:
            local.append(imp)

    return stdlib, installed, local


def resolve_relative_import(relative_module, dotted_folder_path):
    """
    Resolve relative imports based on the current file's dotted folder path.

    Args:
        relative_module: The relative module path (e.g., ".types", "..utils")
        dotted_folder_path: The dotted path of the current file's folder (e.g., "tme", "tme.sub")

    Returns:
        Resolved absolute module path
    """
    if not relative_module.startswith("."):
        return relative_module

    # Count leading dots
    dot_count = 0
    for char in relative_module:
        if char == ".":
            dot_count += 1
        else:
            break

    # Get the module name after the dots
    module_suffix = relative_module[dot_count:]

    # Split the dotted folder path
    if not dotted_folder_path:
        # If we're in the root directory, can't resolve relative imports
        return None

    path_parts = dotted_folder_path.split(".")

    # Calculate the target level
    # 1 dot = same level, 2 dots = go up one level, etc.
    target_level = len(path_parts) - (dot_count - 1)

    if target_level <= 0:
        # Can't go above root level
        return None

    # Build the resolved path
    resolved_parts = path_parts[:target_level]
    if module_suffix:
        resolved_parts.append(module_suffix)

    return ".".join(resolved_parts)


class ASTToJSONParser:
    def __init__(
        self,
        tree,
        source_code: str,
        local_imports: List[str] = None,
        dotted_folder_name: Optional[str] = None,
        requirements_packages: Set[str] = None,
    ):
        self.tree = tree
        self.source_code = source_code.encode("utf-8")
        self.nodes = []
        self.edges = []
        self.current_module = None
        self.current_class = None
        self.global_variables = set()
        self.class_fields = {}
        self.local_imports = set(local_imports) if local_imports else set()
        self.imported_names = {}
        self.imported_aliases = {}
        self.dotted_folder_name = dotted_folder_name
        self.requirements_packages = requirements_packages or set()

    def get_node_text(self, node):
        """Extract text content from a tree-sitter node"""
        return self.source_code[node.start_byte : node.end_byte].decode("utf-8")

    def parse_import_text(self, statement):
        # Normalize multi-line imports by removing newlines and extra whitespace
        normalized = re.sub(r"\s+", " ", statement.strip())

        # Handle parentheses in multi-line imports
        if "(" in normalized and ")" in normalized:
            # Extract content between parentheses and clean it up
            paren_match = re.search(
                r"from\s+(\.+[a-zA-Z_][\w\.]*|[a-zA-Z_][\w\.]*)\s+import\s+\(([^)]+)\)",
                normalized,
            )
            if paren_match:
                module = paren_match.group(1)
                imports_str = paren_match.group(2)
            else:
                return None
        else:
            # Handle single-line imports
            match_from = re.match(
                r"^\s*from\s+(\.+[a-zA-Z_][\w\.]*|[a-zA-Z_][\w\.]*)\s+import\s+(.+)$",
                normalized,
            )
            if not match_from:
                return None
            module = match_from.group(1)
            imports_str = match_from.group(2)
        if module.startswith("."):

            resolved_module = resolve_relative_import(module, self.dotted_folder_name)
            if resolved_module is None:
                print(
                    f" Could not resolve relative import: {module} from {self.dotted_folder_name}"
                )
                return None
            module = resolved_module

        # Parse individual imports
        imports = []
        for imp in [s.strip() for s in imports_str.split(",") if s.strip()]:
            name_alias = re.match(r"^([a-zA-Z_]\w*)(?:\s+as\s+([a-zA-Z_]\w*))?$", imp)
            if name_alias:
                name, alias = name_alias.group(1), name_alias.group(2)
                imports.append({"name": name, "alias": alias})

        return {"type": "from", "module": module, "imports": imports}

    def process_imports(self, node):
        """Process import statements, categorize them, and track local imports"""
        imports_to_categorize = []

        if node.type == "import_statement":
            # Handle: import module [as alias]
            node_text = self.get_node_text(node)
            print(node_text)

            # Parse simple import statement with regex (since parse_import_text only handles 'from' imports)
            import re

            normalized = re.sub(r"\s+", " ", node_text.strip())
            match = re.match(r"^\s*import\s+(.+)$", normalized)
            if match:
                modules_part = match.group(1)
                # Split by comma and handle each module
                for module_spec in modules_part.split(","):
                    module_spec = module_spec.strip()

                    # Check if there's an 'as' alias
                    alias = None
                    if " as " in module_spec:
                        module_name, alias = module_spec.split(" as ", 1)
                        module_name = module_name.strip()
                        alias = alias.strip()
                    else:
                        module_name = module_spec.strip()

                    if module_name:
                        imports_to_categorize.append(module_name)

                        # Categorize this import
                        top_level = module_name.split(".")[0]
                        if top_level in stdlib_modules:
                            # Skip stdlib imports
                            continue
                        elif is_installed_package(
                            top_level, self.requirements_packages
                        ):
                            # Skip installed package imports
                            continue
                        else:
                            # This is a local import - create node
                            display_name = module_name.split(".")[-1]
                            self.add_import_node(display_name, module_name, alias)

        elif node.type == "import_from_statement":
            # Handle: from module import name1, name2 as alias
            print(self.get_node_text(node))
            output = self.parse_import_text(self.get_node_text(node))
            if output is None:
                return

            module_name = output["module"]
            for import_item in output.get("imports", []):
                name = import_item["name"]
                alias = import_item.get("alias")
                full_import_path = f"{module_name}.{name}"
                imports_to_categorize.append(full_import_path)

                # Categorize this import
                top_level = module_name.split(".")[0]
                if top_level in stdlib_modules:
                    # Skip stdlib imports
                    continue
                elif is_installed_package(top_level, self.requirements_packages):
                    # Skip installed package imports
                    continue
                else:
                    # This is a local import - create node
                    self.add_import_node(name, full_import_path, alias)

        for child in node.children:
            self.process_imports(child)

    def is_local_import(self, import_path):
        """Check if an import path is in the local imports list"""
        return import_path in self.local_imports

    def add_import_node(self, name, full_path, alias=None):
        """Add an import node and track the mapping"""
        self.nodes.append(
            {
                "type": "IMPORT",
                "name": name,
                "module": full_path,
                "alias": alias,
                "dotted_folder_name": self.dotted_folder_name,
            }
        )

        # For __init__.py files, create CONTAINS edge from module to import node
        is_init_file = (
            self.dotted_folder_name and self.current_module == self.dotted_folder_name
        ) or (self.current_module == "main" and not self.dotted_folder_name)

        if is_init_file and self.current_module:
            import_display_name = alias if alias else name
            self.edges.append(
                {
                    "type": "CONTAINS",
                    "source": self.current_module,
                    "target": import_display_name,
                    "source_module_name": self.current_module,
                    "target_association_type": "IMPORT",
                }
            )

        # Track the mapping for later usage detection
        display_name = alias if alias else name.split(".")[-1]
        self.imported_names[display_name] = full_path
        if alias:
            self.imported_aliases[alias] = name

    def is_main_block(self, if_node):
        """Check if an if_statement is the if __name__ == "__main__": pattern"""
        # Look for the condition: __name__ == "__main__"
        for child in if_node.children:
            if child.type == "comparison_operator":
                # Check if it's comparing __name__ with "__main__"
                left_operand = None
                right_operand = None

                for comparison_child in child.children:
                    if comparison_child.type == "identifier":
                        text = self.get_node_text(comparison_child)
                        if text == "__name__":
                            left_operand = text
                    elif comparison_child.type == "string":
                        text = self.get_node_text(comparison_child)
                        if text in ['"__main__"', "'__main__'"]:
                            right_operand = text

                if left_operand == "__name__" and right_operand:
                    return True
        return False

    def get_signature(self, node):
        """Extract signature from function/method/class definition"""
        if node.type in ["function_definition", "async_function_definition"]:
            # Find the signature part (everything before the colon)
            for child in node.children:
                if child.type == ":":
                    break
                if child.type in ["def", "async"]:
                    continue
            # Get from 'def' to ':'
            start = node.start_byte
            for child in node.children:
                if child.type == ":":
                    end = child.start_byte
                    break
            else:
                end = node.end_byte
            return self.source_code[start:end].decode("utf-8").strip()

        elif node.type == "class_definition":
            # Get from 'class' to ':'
            start = node.start_byte
            for child in node.children:
                if child.type == ":":
                    end = child.start_byte
                    break
            else:
                end = node.end_byte
            return self.source_code[start:end].decode("utf-8").strip()

        return ""

    def extract_name(self, node):
        """Extract name from definition nodes"""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_text(child)
        return ""

    def find_global_variable_usage(self, assignment_node, var_name):
        """Find what a global variable uses in its assignment"""
        used_classes = set()
        used_functions = set()
        used_imports = set()
        used_global_vars = set()

        def traverse_assignment(node):
            if node.type == "call":
                # Handle function/class calls
                if node.children and node.children[0].type == "identifier":
                    called_name = self.get_node_text(node.children[0])
                    if called_name in self.imported_names:
                        used_imports.add(called_name)
                    elif self.is_class_usage(called_name):
                        used_classes.add(called_name)
                    elif self.is_function_defined(called_name):
                        used_functions.add(called_name)

                elif node.children and node.children[0].type == "attribute":
                    # Handle attribute calls like module.Class()
                    attr_node = node.children[0]
                    if (
                        len(attr_node.children) >= 2
                        and attr_node.children[0].type == "identifier"
                    ):
                        base_name = self.get_node_text(attr_node.children[0])
                        if base_name in self.imported_names:
                            used_imports.add(base_name)

                for child in node.children:
                    if child.type == "argument_list":
                        self.process_call_arguments(
                            child,
                            used_classes,
                            used_functions,
                            used_imports,
                            used_global_vars,
                        )
            elif node.type == "identifier":
                # Handle direct identifier usage
                name = self.get_node_text(node)
                if name in self.imported_names:
                    used_imports.add(name)
                elif self.is_class_usage(name):
                    used_classes.add(name)
                elif self.is_function_defined(name):
                    used_functions.add(name)
                elif name in self.global_variables:
                    used_global_vars.add(name)
            elif node.type == "attribute":
                # Handle attribute access like module.Class
                if len(node.children) >= 2 and node.children[0].type == "identifier":
                    base_name = self.get_node_text(node.children[0])
                    if base_name in self.imported_names:
                        used_imports.add(base_name)

            for child in node.children:
                traverse_assignment(child)

        # Only traverse the right side of the assignment
        for child in assignment_node.children:
            if child.type not in ["identifier", "pattern_list", "tuple_pattern"]:
                traverse_assignment(child)

        # Create USES edges from global variable to what it uses
        for class_name in used_classes:
            class_signature = None
            class_module = None
            for node in self.nodes:
                if node.get("type") == "Class" and node.get("name") == class_name:
                    class_signature = node.get("signature")
                    class_module = node.get("module_name")
                    break

            self.edges.append(
                {
                    "type": "USES",
                    "source": var_name,
                    "target": class_name,
                    "source_association_type": "GlobalVariable",
                    "target_association_type": "Class",
                    "source_module_name": self.current_module,
                    "target_module_name": class_module or self.current_module,
                    "target_signature": class_signature,
                }
            )

        for func_name in used_functions:
            func_signature = None
            func_module = None
            for node in self.nodes:
                if node.get("type") == "Function" and node.get("name") == func_name:
                    func_signature = node.get("signature")
                    func_module = node.get("module_name")
                    break

            self.edges.append(
                {
                    "type": "USES",
                    "source": var_name,
                    "target": func_name,
                    "source_association_type": "GlobalVariable",
                    "target_association_type": "Function",
                    "source_module_name": self.current_module,
                    "target_module_name": func_module or self.current_module,
                    "target_signature": func_signature,
                }
            )

        for import_name in used_imports:
            self.edges.append(
                {
                    "type": "USES",
                    "source": var_name,
                    "target": import_name,
                    "source_association_type": "GlobalVariable",
                    "target_association_type": "IMPORT",
                    "source_module_name": self.current_module,
                }
            )
        for global_var_name in used_global_vars:
            self.edges.append(
                {
                    "type": "USES",
                    "source": var_name,
                    "target": global_var_name,
                    "source_association_type": "GlobalVariable",
                    "target_association_type": "GlobalVariable",
                    "source_module_name": self.current_module,
                    "target_module_name": self.current_module,
                }
            )

    def find_list_members_usage(
        self,
        list_node,
        used_classes,
        used_functions,
        used_imports,
        used_global_vars=None,
    ):
        """Extract identifiers from list literals and categorize them"""
        if used_global_vars is None:
            used_global_vars = set()

        def extract_identifiers_from_list(node):
            """Recursively extract identifiers from list and nested structures"""
            if node.type == "identifier":
                name = self.get_node_text(node)
                print(f"Found list member: {name}")

                # Categorize the identifier
                if name in self.imported_names:
                    used_imports.add(name)
                    print(f"  -> Categorized as import: {name}")
                elif self.is_class_usage(name):
                    used_classes.add(name)
                    print(f"  -> Categorized as class: {name}")
                elif self.is_function_defined(name):
                    used_functions.add(name)
                    print(f"  -> Categorized as function: {name}")
                elif name in self.global_variables:
                    used_global_vars.add(name)
                    print(f"  -> Categorized as global variable: {name}")
                else:
                    print(f"  -> Unknown identifier: {name}")

            elif node.type == "attribute":
                # Handle module.Class references in lists
                if len(node.children) >= 2 and node.children[0].type == "identifier":
                    base_name = self.get_node_text(node.children[0])
                    if base_name in self.imported_names:
                        used_imports.add(base_name)
                        print(f"  -> Found module reference in list: {base_name}")

            # Recursively process children
            for child in node.children:
                extract_identifiers_from_list(child)

        # Extract all identifiers from the list
        extract_identifiers_from_list(list_node)

    def is_function_defined(self, name):
        """Check if a name refers to a defined function"""
        for node in self.nodes:
            if node.get("type") == "Function" and node.get("name") == name:
                return True
        return False

    def find_global_variables(self, node):
        """Find all global variable assignments"""
        if node.type == "assignment":
            # Recursively heck if this assignment is at the module level
            parent = node.parent
            while parent and parent.type != "module":
                if parent.type in [
                    "function_definition",
                    "async_function_definition",
                    "class_definition",
                    "if_statement",
                    "for_statement",
                    "while_statement",
                    "with_statement",
                    "try_statement",
                ]:
                    return
                elif parent.type == "if_statement":
                    # Allow if it's the __main__ block, otherwise exclude
                    if self.is_main_block(parent):
                        is_in_main_block = True
                        break
                    else:
                        return
                parent = parent.parent

            # Extract variable names from the left side of assignment
            for child in node.children:
                if child.type in ["identifier", "pattern_list", "tuple_pattern"]:
                    if child.type == "identifier":
                        var_name = self.get_node_text(child)
                        self.global_variables.add(var_name)
                        self.nodes.append(
                            {
                                "type": "GlobalVariable",
                                "name": var_name,
                                "code": self.get_node_text(node),
                                "module_name": self.current_module,
                            }
                        )
                        if self.current_module:
                            self.edges.append(
                                {
                                    "type": "CONTAINS",
                                    "source": self.current_module,
                                    "target": var_name,
                                    "source_module_name": self.current_module,
                                    "target_module_name": self.current_module,
                                }
                            )

                        # Find what this global variable uses
                        self.find_global_variable_usage(node, var_name)

                    elif child.type in ["pattern_list", "tuple_pattern"]:
                        for grandchild in child.children:
                            if grandchild.type == "identifier":
                                var_name = self.get_node_text(grandchild)
                                self.global_variables.add(var_name)
                                self.nodes.append(
                                    {
                                        "type": "GlobalVariable",
                                        "name": var_name,
                                        "code": self.get_node_text(node),
                                        "module_name": self.current_module,
                                    }
                                )
                                if self.current_module:
                                    self.edges.append(
                                        {
                                            "type": "CONTAINS",
                                            "source": self.current_module,
                                            "target": var_name,
                                            "source_module_name": self.current_module,
                                            "target_module_name": self.current_module,
                                        }
                                    )

                                # Find what this global variable uses
                                self.find_global_variable_usage(node, var_name)
                    break

        # Recursively check children
        for child in node.children:
            self.find_global_variables(child)

    def find_class_fields(self, class_node, class_name):
        """Find field assignments within a class"""
        fields = set()

        def traverse_class_body(node):
            if node.type == "assignment":
                # Look for self.field_name assignments
                for child in node.children:
                    if (
                        child.type == "attribute"
                        and len(child.children) >= 2
                        and child.children[0].type == "identifier"
                        and self.get_node_text(child.children[0]) == "self"
                    ):
                        field_name = self.get_node_text(
                            child.children[2]
                        )  # after the dot
                        fields.add(field_name)
                        break
            # Also look for simple assignments that might be class variables
            elif node.type == "expression_statement":
                for child in node.children:
                    if child.type == "assignment":
                        left_side = child.children[0] if child.children else None
                        if left_side and left_side.type == "identifier":
                            # This could be a class variable
                            field_name = self.get_node_text(left_side)
                            fields.add(field_name)

            for child in node.children:
                traverse_class_body(child)

        traverse_class_body(class_node)
        return fields

    def find_class_field_usage(self, class_node, class_name):
        """Find what class-level fields use in their assignments and create USES edges from Class to dependencies"""
        used_classes = set()
        used_functions = set()
        used_imports = set()
        used_global_vars = set()

        def traverse_assignment(node):
            if node.type == "call":
                # Handle function/class calls
                if node.children and node.children[0].type == "identifier":
                    # print("called_name", self.get_node_text(node.children[0]))
                    called_name = self.get_node_text(node.children[0])
                    if called_name in self.imported_names:
                        # print("called_import", called_name)
                        used_imports.add(called_name)
                    elif self.is_class_usage(called_name):
                        used_classes.add(called_name)
                    elif self.is_function_defined(called_name):
                        used_functions.add(called_name)
                elif node.children and node.children[0].type == "attribute":
                    # Handle attribute calls like module.Class()
                    attr_node = node.children[0]
                    if (
                        len(attr_node.children) >= 2
                        and attr_node.children[0].type == "identifier"
                    ):
                        base_name = self.get_node_text(attr_node.children[0])
                        if base_name in self.imported_names:
                            used_imports.add(base_name)
                for child in node.children:
                    if child.type == "argument_list":
                        self.process_call_arguments(
                            child,
                            used_classes,
                            used_functions,
                            used_imports,
                            used_global_vars,
                        )
            elif node.type == "identifier":
                # Handle direct identifier usage
                name = self.get_node_text(node)
                if name in self.imported_names:
                    used_imports.add(name)
                elif self.is_class_usage(name):
                    used_classes.add(name)
                elif self.is_function_defined(name):
                    used_functions.add(name)
                elif name in self.global_variables:
                    used_global_vars.add(name)
            elif node.type == "attribute":
                # Handle attribute access like module.Class
                if len(node.children) >= 2 and node.children[0].type == "identifier":
                    base_name = self.get_node_text(node.children[0])
                    if base_name in self.imported_names:
                        used_imports.add(base_name)

            for child in node.children:
                traverse_assignment(child)

        def traverse_class_body(node):
            if node.type == "assignment":
                # Only traverse the right side of the assignment
                for child in node.children:
                    if child.type not in [
                        "identifier",
                        "pattern_list",
                        "tuple_pattern",
                    ]:
                        traverse_assignment(child)
            # Also check expression statements that might contain assignments
            elif node.type == "expression_statement":
                for child in node.children:
                    if child.type == "assignment":
                        # Only traverse the right side of the assignment
                        for assign_child in child.children:
                            if assign_child.type not in [
                                "identifier",
                                "pattern_list",
                                "tuple_pattern",
                            ]:
                                traverse_assignment(assign_child)

            for child in node.children:
                traverse_class_body(child)

        # Traverse the class body to find field assignments
        traverse_class_body(class_node)

        # Get class signature for source identification
        class_signature = self.get_signature(class_node)

        # Create USES edges from Class to what its fields use
        for class_used in used_classes:
            class_used_signature = None
            class_used_module = None
            for node in self.nodes:
                if node.get("type") == "Class" and node.get("name") == class_used:
                    class_used_signature = node.get("signature")
                    class_used_module = node.get("module_name")
                    break

            self.edges.append(
                {
                    "type": "USES",
                    "source": class_name,
                    "target": class_used,
                    "source_association_type": "Class",
                    "target_association_type": "Class",
                    "source_module_name": self.current_module,
                    "source_signature": class_signature,
                    "target_module_name": class_used_module or self.current_module,
                    "target_signature": class_used_signature,
                }
            )

        for func_used in used_functions:
            func_used_signature = None
            func_used_module = None
            for node in self.nodes:
                if node.get("type") == "Function" and node.get("name") == func_used:
                    func_used_signature = node.get("signature")
                    func_used_module = node.get("module_name")
                    break

            self.edges.append(
                {
                    "type": "USES",
                    "source": class_name,
                    "target": func_used,
                    "source_association_type": "Class",
                    "target_association_type": "Function",
                    "source_module_name": self.current_module,
                    "source_signature": class_signature,
                    "target_module_name": func_used_module or self.current_module,
                    "target_signature": func_used_signature,
                }
            )

        for import_used in used_imports:
            print("import_used", import_used)
            self.edges.append(
                {
                    "type": "USES",
                    "source": class_name,
                    "target": import_used,
                    "source_association_type": "Class",
                    "target_association_type": "IMPORT",
                    "source_module_name": self.current_module,
                    "source_signature": class_signature,
                }
            )

        for global_var_used in used_global_vars:
            self.edges.append(
                {
                    "type": "USES",
                    "source": class_name,
                    "target": global_var_used,
                    "source_association_type": "Class",
                    "target_association_type": "GlobalVariable",
                    "source_module_name": self.current_module,
                    "source_signature": class_signature,
                    "target_module_name": self.current_module,
                }
            )

    def find_variable_usage(self, func_node, func_name, is_method=False):
        """Find variables used within a function or method"""
        used_vars = set()
        used_classes = set()
        used_functions = set()
        used_imports = set()

        def traverse_function(node):
            if node.type == "identifier":
                var_name = self.get_node_text(node)
                # Check if it's a local import
                if var_name in self.imported_names:

                    used_imports.add(var_name)
                # Check if it's a known class
                elif self.is_class_usage(var_name):
                    used_classes.add(var_name)
                # Check if it's a function call
                elif self.is_function_usage(node, var_name):
                    used_functions.add(var_name)
                else:
                    used_vars.add(var_name)
            elif node.type == "attribute":
                # Handle self.field_name and other attribute access
                if (
                    len(node.children) >= 2
                    and node.children[0].type == "identifier"
                    and self.get_node_text(node.children[0]) == "self"
                ):
                    field_name = self.get_node_text(node.children[2])
                    used_vars.add(f"self.{field_name}")
                else:
                    # Handle other attribute access like module.function
                    base_name = (
                        self.get_node_text(node.children[0]) if node.children else ""
                    )
                    if base_name in self.imported_names:
                        used_imports.add(base_name)
            elif node.type == "call":
                # Handle function calls - check the function being called
                if node.children and node.children[0].type == "identifier":
                    func_called = self.get_node_text(node.children[0])
                    if func_called in self.imported_names:
                        used_imports.add(func_called)
                    elif self.is_class_usage(func_called):
                        used_classes.add(func_called)
                    elif self.is_function_defined(func_called):
                        used_functions.add(func_called)
                for child in node.children:
                    if child.type == "argument_list":
                        self.process_call_arguments(
                            child, used_classes, used_functions, used_imports, used_vars
                        )

            for child in node.children:
                traverse_function(child)

        traverse_function(func_node)

        # Get function signature for source identification
        func_signature = self.get_signature(func_node)

        # Create USES edges for different types
        for var in used_vars:
            if var.startswith("self.") and is_method:
                field_name = var[5:]  # Remove 'self.'
                if self.current_class and field_name in self.class_fields.get(
                    self.current_class, set()
                ):
                    self.edges.append(
                        {
                            "type": "USES",
                            "source": func_name,
                            "target": field_name,
                            "source_association_type": "Method",
                            "target_association_type": "Field",
                            "source_module_name": self.current_module,
                            "source_signature": func_signature,
                            "source_class": self.current_class,
                            "target_module_name": self.current_module,
                            "target_class": self.current_class,
                        }
                    )
            elif var in self.global_variables:
                self.edges.append(
                    {
                        "type": "USES",
                        "source": func_name,
                        "target": var,
                        "source_association_type": (
                            "Method" if is_method else "Function"
                        ),
                        "target_association_type": "GlobalVariable",
                        "source_module_name": self.current_module,
                        "source_signature": func_signature,
                        "source_class": self.current_class if is_method else None,
                        "target_module_name": self.current_module,
                    }
                )

        # Create USES edges for classes
        for class_name in used_classes:
            # Find the class node to get its signature
            class_signature = None
            class_module = None
            for node in self.nodes:
                if node.get("type") == "Class" and node.get("name") == class_name:
                    class_signature = node.get("signature")
                    class_module = node.get("module_name")
                    break

            self.edges.append(
                {
                    "type": "USES",
                    "source": func_name,
                    "target": class_name,
                    "source_association_type": "Method" if is_method else "Function",
                    "target_association_type": "Class",
                    "source_module_name": self.current_module,
                    "source_signature": func_signature,
                    "source_class": self.current_class if is_method else None,
                    "target_module_name": class_module or self.current_module,
                    "target_signature": class_signature,
                }
            )

        # Create USES edges for functions
        for func_used in used_functions:
            # Find the function node to get its signature
            func_used_signature = None
            func_used_module = None
            for node in self.nodes:
                if node.get("type") == "Function" and node.get("name") == func_used:
                    func_used_signature = node.get("signature")
                    func_used_module = node.get("module_name")
                    break

            self.edges.append(
                {
                    "type": "USES",
                    "source": func_name,
                    "target": func_used,
                    "source_association_type": "Method" if is_method else "Function",
                    "target_association_type": "Function",
                    "source_module_name": self.current_module,
                    "source_signature": func_signature,
                    "source_class": self.current_class if is_method else None,
                    "target_module_name": func_used_module or self.current_module,
                    "target_signature": func_used_signature,
                }
            )

        # Create USES edges for imports
        for import_name in used_imports:
            self.edges.append(
                {
                    "type": "USES",
                    "source": func_name,
                    "target": import_name,
                    "source_association_type": "Method" if is_method else "Function",
                    "target_association_type": "IMPORT",
                    "source_module_name": self.current_module,
                    "source_signature": func_signature,
                    "source_class": self.current_class if is_method else None,
                }
            )

    def process_call_arguments(
        self,
        argument_list_node,
        used_classes,
        used_functions,
        used_imports,
        used_global_vars,
    ):
        """Process arguments in function calls to detect dependencies"""

        def traverse_arguments(node):
            if node.type == "identifier":
                arg_name = self.get_node_text(node)
                # Categorize the argument
                if arg_name in self.imported_names:
                    used_imports.add(arg_name)
                    print(f"  -> Found import argument: {arg_name}")
                elif self.is_class_usage(arg_name):
                    used_classes.add(arg_name)
                    print(f"  -> Found class argument: {arg_name}")
                elif self.is_function_defined(arg_name):
                    used_functions.add(arg_name)
                    print(f"  -> Found function argument: {arg_name}")
                elif arg_name in self.global_variables:
                    used_global_vars.add(arg_name)
                    print(f"  -> Found global variable argument: {arg_name}")

            elif node.type == "keyword_argument":
                # Handle keyword arguments like func=web_search
                for child in node.children:
                    if child.type == "identifier":
                        # This could be the value of a keyword argument
                        arg_name = self.get_node_text(child)
                        if arg_name in self.imported_names:
                            used_imports.add(arg_name)
                            print(f"  -> Found import in keyword arg: {arg_name}")
                        elif self.is_class_usage(arg_name):
                            used_classes.add(arg_name)
                            print(f"  -> Found class in keyword arg: {arg_name}")
                        elif self.is_function_defined(arg_name):
                            used_functions.add(arg_name)
                            print(f"  -> Found function in keyword arg: {arg_name}")
                        elif arg_name in self.global_variables:
                            used_global_vars.add(arg_name)
                            print(f"  -> Found global var in keyword arg: {arg_name}")

            elif node.type == "attribute":
                # Handle attribute access in arguments like obj.method
                if len(node.children) >= 2 and node.children[0].type == "identifier":
                    base_name = self.get_node_text(node.children[0])
                    if base_name in self.imported_names:
                        used_imports.add(base_name)
                        print(f"  -> Found import attribute in arg: {base_name}")

            # Recursively process child nodes
            for child in node.children:
                traverse_arguments(child)

        # Process all arguments
        traverse_arguments(argument_list_node)

    def is_class_usage(self, name):
        """Check if a name refers to a class"""
        # Check if it's an imported class
        # if name in self.imported_names and "class" in self.imported_names[name].lower():
        #     return True
        # Check if it's a locally defined class
        for node in self.nodes:
            if node.get("type") == "Class" and node.get("name") == name:
                return True
        return False

    def is_function_usage(self, node, name):
        """Check if a name refers to a function call"""
        # Look at the parent to see if this identifier is being called
        parent = node.parent if hasattr(node, "parent") else None
        if parent and parent.type == "call":
            return True
        return False

    def discover_all_classes(self, node):
        """First pass: discover all class definitions"""
        classes_found = []

        def traverse_for_classes(node):
            if node.type == "class_definition":
                class_name = self.extract_name(node)
                class_signature = self.get_signature(node)

                classes_found.append(
                    {
                        "name": class_name,
                        "signature": class_signature,
                        "code": self.get_node_text(node),
                    }
                )

            for child in node.children:
                traverse_for_classes(child)

        traverse_for_classes(node)

        # Create class nodes using the combined function
        for class_info in classes_found:
            self.create_or_update_class_node(class_info)

        return classes_found

    def discover_all_functions(self, node):
        """First pass: discover all function definitions"""
        functions_found = []

        def traverse_for_functions(node, current_class=None):
            if node.type in ["function_definition", "async_function_definition"]:
                func_name = self.extract_name(node)
                func_signature = self.get_signature(node)
                is_method = current_class is not None

                functions_found.append(
                    {
                        "name": func_name,
                        "signature": func_signature,
                        "is_method": is_method,
                        "class": current_class,
                        "code": self.get_node_text(node),
                    }
                )

            elif node.type == "class_definition":
                class_name = self.extract_name(node)
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            traverse_for_functions(block_child, class_name)

            for child in node.children:
                if node.type != "class_definition":
                    traverse_for_functions(child, current_class)

        traverse_for_functions(node)

        # Create function nodes using the combined function
        for func_info in functions_found:
            self.create_or_update_function_node(func_info)

        return functions_found

    def discover_all_definitions(self, node):
        """Combined function to discover all classes and functions (nodes only)"""
        print("🔍 Discovering all class and function definitions...")

        # First discover all classes
        classes_found = self.discover_all_classes(node)
        print(f"Found {len(classes_found)} classes")

        # Then discover all functions (including methods)
        functions_found = self.discover_all_functions(node)
        print(f"Found {len(functions_found)} functions/methods")

        # No relationship creation here - that happens in normal processing
        return classes_found, functions_found

    def find_existing_node(self, node_type, name, signature=None, class_name=None):
        """Find an existing node in self.nodes based on type, name, and signature"""
        for node in self.nodes:
            if node.get("type") == node_type and node.get("name") == name:
                # For methods, also check class name
                if node_type == "Method":
                    if node.get("class") == class_name:
                        return node
                # For functions and classes, check signature if provided
                elif signature is None or node.get("signature") == signature:
                    return node
        return None

    def create_or_update_class_node(self, class_info):
        """Create or update a class node (handles both discovery and normal processing)"""
        class_name = class_info["name"]
        class_signature = class_info["signature"]

        existing_node = self.find_existing_node("Class", class_name, class_signature)

        if existing_node:
            # Update existing node with any new information
            existing_node.update(
                {
                    "code": class_info["code"],
                    "signature": class_signature,
                    "module_name": self.current_module,
                }
            )
            print(f"Updated existing class: {class_name}")
        else:
            # Create new node
            new_node = {
                "type": "Class",
                "name": class_name,
                "signature": class_signature,
                "code": class_info["code"],
                "module_name": self.current_module,
            }
            self.nodes.append(new_node)
            print(f"Created new class: {class_name}")

    def create_or_update_function_node(self, func_info):
        """Create or update a function node (handles both discovery and normal processing)"""
        func_name = func_info["name"]
        func_signature = func_info["signature"]
        is_method = func_info.get("is_method", False)
        class_name = func_info.get("class", None)

        existing_node = self.find_existing_node(
            "Method" if is_method else "Function", func_name, func_signature, class_name
        )

        if existing_node:
            # Update existing node with any new information
            existing_node.update(
                {
                    "code": func_info["code"],
                    "signature": func_signature,
                    "module_name": self.current_module,
                }
            )
            if is_method and class_name:
                existing_node["class"] = class_name
            print(
                f"Updated existing {'method' if is_method else 'function'}: {func_name}"
            )
        else:
            # Create new node
            if is_method:
                new_node = {
                    "type": "Method",
                    "name": func_name,
                    "class": class_name,
                    "code": func_info["code"],
                    "signature": func_signature,
                    "module_name": self.current_module,
                }
            else:
                new_node = {
                    "type": "Function",
                    "name": func_name,
                    "code": func_info["code"],
                    "signature": func_signature,
                    "module_name": self.current_module,
                }
            self.nodes.append(new_node)
            print(f"Created new {'method' if is_method else 'function'}: {func_name}")

    def process_node(self, node):
        """Process a single AST node"""
        if node.type == "module":
            # Create module node
            module_name = getattr(self, "module_name", "main")
            self.current_module = module_name
            local_name = module_name.split(".")[-1]
            self.nodes.append(
                {"type": "MODULE", "name": module_name, "local_name": local_name}
            )

            # First pass: find all imports
            self.process_imports(node)
            # second pass: find all class and function definitions
            self.discover_all_definitions(node)
            # thir pass: find all global variables
            self.find_global_variables(node)

            # Process children
            for child in node.children:
                if child.type not in ["import_statement", "import_from_statement"]:
                    self.process_node(child)

        # Handle class definitions - create class nodes and track class context
        elif node.type == "class_definition":
            class_name = self.extract_name(node)
            class_signature = self.get_signature(node)
            prev_class = self.current_class  # Save current class context
            self.current_class = class_name

            # Find class fields
            fields = self.find_class_fields(node, class_name)
            self.class_fields[class_name] = fields

            # Create class node
            class_info = {
                "name": class_name,
                "signature": class_signature,
                "code": self.get_node_text(node),
                "fields": fields,  # Additional info for normal processing
            }
            self.create_or_update_class_node(class_info)

            # Create CONTAINS edge from module to class
            if self.current_module:
                self.edges.append(
                    {
                        "type": "CONTAINS",
                        "source": self.current_module,
                        "target": class_name,
                        "source_module_name": self.current_module,
                        "target_module_name": self.current_module,
                        "target_signature": class_signature,
                    }
                )
            self.find_class_field_usage(node, class_name)
            # Create field nodes and HAS_FIELD edges
            for field_name in fields:
                self.nodes.append(
                    {
                        "type": "Field",
                        "name": field_name,
                        "class": class_name,
                        "module_name": self.current_module,
                    }
                )
                self.edges.append(
                    {
                        "type": "HAS_FIELD",
                        "source": class_name,
                        "target": field_name,
                        "source_module_name": self.current_module,
                        "source_signature": class_signature,
                        "target_module_name": self.current_module,
                        "target_class": class_name,
                    }
                )

            # Handle inheritance
            for child in node.children:
                if child.type == "argument_list":
                    for arg_child in child.children:
                        if arg_child.type == "identifier":
                            base_class = self.get_node_text(arg_child)
                            # Find parent class signature
                            parent_signature = None
                            parent_module = None
                            found_local = False
                            for node_data in self.nodes:
                                if (
                                    node_data.get("type") == "Class"
                                    and node_data.get("name") == base_class
                                ):
                                    parent_signature = node_data.get("signature")
                                    parent_module = node_data.get("module_name")
                                    found_local = True
                                    break

                            if not found_local:
                                is_imported_parent = False
                                if base_class in self.imported_names:
                                    # It's an imported class - create INHERITS to Import node
                                    parent_module = self.imported_names[base_class]
                                    is_imported_parent = True
                                    print(
                                        f"Found imported base class: {base_class} from {parent_module}"
                                    )
                                else:
                                    # Check Import nodes directly (in case of aliases)
                                    for node_data in self.nodes:
                                        if node_data.get("type") == "IMPORT" and (
                                            node_data.get("name") == base_class
                                            or node_data.get("alias") == base_class
                                        ):
                                            parent_module = node_data.get("module")
                                            is_imported_parent = True
                                            print(
                                                f"Found imported base class via Import node: {base_class} from {parent_module}"
                                            )
                                            break

                                if is_imported_parent:
                                    # Create INHERITS relationship to Import node (will be redirected during import merging)
                                    self.edges.append(
                                        {
                                            "type": "INHERITS",
                                            "source": class_name,
                                            "target": base_class,
                                            "source_module_name": self.current_module,
                                            "source_signature": class_signature,
                                            "target_association_type": "IMPORT",  # Indicates this targets an Import node
                                        }
                                    )
                                else:
                                    # Unknown parent class - create relationship anyway for potential future resolution
                                    self.edges.append(
                                        {
                                            "type": "INHERITS",
                                            "source": class_name,
                                            "target": base_class,
                                            "source_module_name": self.current_module,
                                            "source_signature": class_signature,
                                            "target_module_name": self.current_module,  # Default to current module
                                            "target_signature": None,
                                        }
                                    )
                            else:
                                # Local parent class - use existing logic
                                self.edges.append(
                                    {
                                        "type": "INHERITS",
                                        "source": class_name,
                                        "target": base_class,
                                        "source_module_name": self.current_module,
                                        "source_signature": class_signature,
                                        "target_module_name": parent_module
                                        or self.current_module,
                                        "target_signature": parent_signature,
                                    }
                                )

            # Process class body (methods, nested classes, etc.)
            for child in node.children:
                if child.type == "block":
                    for block_child in child.children:
                        self.process_node(block_child)

            # Restore previous class context
            self.current_class = prev_class

        elif node.type in ["function_definition", "async_function_definition"]:
            func_name = self.extract_name(node)
            func_signature = self.get_signature(node)
            is_method = self.current_class is not None

            if is_method:
                # Create method node
                func_info = {
                    "name": func_name,
                    "signature": func_signature,
                    "code": self.get_node_text(node),
                    "is_method": is_method,
                    "class": self.current_class,
                }
                self.create_or_update_function_node(func_info)

                # Create HAS_METHOD edge
                # Find class signature
                class_signature = None
                for node_data in self.nodes:
                    if (
                        node_data.get("type") == "Class"
                        and node_data.get("name") == self.current_class
                    ):
                        class_signature = node_data.get("signature")
                        break

                self.edges.append(
                    {
                        "type": "HAS_METHOD",
                        "source": self.current_class,
                        "target": func_name,
                        "source_module_name": self.current_module,
                        "source_signature": class_signature,
                        "target_module_name": self.current_module,
                        "target_signature": func_signature,
                        "target_class": self.current_class,
                    }
                )
            else:
                # Create function node
                func_info = {
                    "name": func_name,
                    "signature": func_signature,
                    "code": self.get_node_text(node),
                    "is_method": is_method,
                    "class": self.current_class,
                }
                self.create_or_update_function_node(func_info)

                # Create CONTAINS edge from module to function
                if self.current_module:
                    self.edges.append(
                        {
                            "type": "CONTAINS",
                            "source": self.current_module,
                            "target": func_name,
                            "source_module_name": self.current_module,
                            "target_module_name": self.current_module,
                            "target_signature": func_signature,
                        }
                    )

            # Find variable usage
            self.find_variable_usage(node, func_name, is_method)

            # Process function body if needed (for nested functions)
            for child in node.children:
                if child.type == "block":
                    for block_child in child.children:
                        if block_child.type in [
                            "function_definition",
                            "async_function_definition",
                            "class_definition",
                        ]:
                            self.process_node(block_child)

        elif node.type == "if_statement":
            # Check if this is the special if __name__ == "__main__": block
            if self.is_main_block(node):
                func_name = "__main__"
                # Create a synthetic signature for the main block
                func_signature = 'if __name__ == "__main__":'

                # Create function node for the main block
                func_node = {
                    "type": "Function",
                    "name": func_name,
                    "code": self.get_node_text(node),
                    "signature": func_signature,
                    "module_name": self.current_module,
                }
                self.nodes.append(func_node)

                # Create CONTAINS edge from module to main function
                if self.current_module:
                    self.edges.append(
                        {
                            "type": "CONTAINS",
                            "source": self.current_module,
                            "target": func_name,
                            "source_module_name": self.current_module,
                            "target_module_name": self.current_module,
                            "target_signature": func_signature,
                        }
                    )

                # Process the main block body for nested functions/classes
                for child in node.children:
                    if child.type == "block":
                        self.find_variable_usage_in_block(
                            child, func_name, is_method=False
                        )

                        for block_child in child.children:
                            if block_child.type in [
                                "function_definition",
                                "async_function_definition",
                                "class_definition",
                            ]:
                                self.process_node(block_child)
        else:
            # For other node types, continue processing children
            for child in node.children:
                if child.type not in ["import_statement", "import_from_statement"]:
                    self.process_node(child)

    def find_variable_usage_in_block(self, block_node, func_name, is_method=False):
        """Find variables used within a block (for __main__ blocks)"""
        used_vars = set()
        used_classes = set()
        used_functions = set()
        used_imports = set()

        def traverse_block(node):
            if node.type == "identifier":
                var_name = self.get_node_text(node)
                # Check if it's a local import
                if var_name in self.imported_names:
                    used_imports.add(var_name)
                # Check if it's a known class
                elif self.is_class_usage(var_name):
                    used_classes.add(var_name)
                # Check if it's a function call
                elif self.is_function_usage(node, var_name):
                    used_functions.add(var_name)
                else:
                    used_vars.add(var_name)
            elif node.type == "attribute":
                # Handle attribute access
                if len(node.children) >= 2 and node.children[0].type == "identifier":
                    base_name = self.get_node_text(node.children[0])
                    if base_name in self.imported_names:
                        used_imports.add(base_name)
            elif node.type == "call":
                # Handle function calls - check the function being called
                if node.children and node.children[0].type == "identifier":
                    func_called = self.get_node_text(node.children[0])
                    if func_called in self.imported_names:
                        used_imports.add(func_called)
                    elif self.is_class_usage(func_called):
                        used_classes.add(func_called)
                    elif self.is_function_defined(func_called):
                        used_functions.add(func_called)
                for child in node.children:
                    if child.type == "argument_list":
                        self.process_call_arguments(
                            child, used_classes, used_functions, used_imports, used_vars
                        )

            for child in node.children:
                traverse_block(child)

        traverse_block(block_node)

        # Create the same edges as in find_variable_usage
        func_signature = 'if __name__ == "__main__":'

        # Create USES edges for different types
        for var in used_vars:
            if var in self.global_variables:
                self.edges.append(
                    {
                        "type": "USES",
                        "source": func_name,
                        "target": var,
                        "source_association_type": "Function",
                        "target_association_type": "GlobalVariable",
                        "source_module_name": self.current_module,
                        "source_signature": func_signature,
                        "target_module_name": self.current_module,
                    }
                )

        # Create USES edges for classes
        for class_name in used_classes:
            class_signature = None
            class_module = None
            for node in self.nodes:
                if node.get("type") == "Class" and node.get("name") == class_name:
                    class_signature = node.get("signature")
                    class_module = node.get("module_name")
                    break

            self.edges.append(
                {
                    "type": "USES",
                    "source": func_name,
                    "target": class_name,
                    "source_association_type": "Function",
                    "target_association_type": "Class",
                    "source_module_name": self.current_module,
                    "source_signature": func_signature,
                    "target_module_name": class_module or self.current_module,
                    "target_signature": class_signature,
                }
            )

        # Create USES edges for functions
        for func_used in used_functions:
            func_used_signature = None
            func_used_module = None
            for node in self.nodes:
                if node.get("type") == "Function" and node.get("name") == func_used:
                    func_used_signature = node.get("signature")
                    func_used_module = node.get("module_name")
                    break

            self.edges.append(
                {
                    "type": "USES",
                    "source": func_name,
                    "target": func_used,
                    "source_association_type": "Function",
                    "target_association_type": "Function",
                    "source_module_name": self.current_module,
                    "source_signature": func_signature,
                    "target_module_name": func_used_module or self.current_module,
                    "target_signature": func_used_signature,
                }
            )

        # Create USES edges for imports
        for import_name in used_imports:
            self.edges.append(
                {
                    "type": "USES",
                    "source": func_name,
                    "target": import_name,
                    "source_association_type": "Function",
                    "target_association_type": "IMPORT",
                    "source_module_name": self.current_module,
                    "source_signature": func_signature,
                }
            )

    def parse(self, module_name: str = "main") -> Dict[str, Any]:
        """Parse the AST and return JSON representation"""
        self.module_name = module_name
        self.process_node(self.tree.root_node)

        return {"nodes": self.nodes, "edges": self.edges}


def parse_code_to_json(
    code: str,
    parser,
    module_name: str = "main",
    local_imports: List[str] = None,
    dotted_folder_name: Optional[str] = None,
    requirements_packages: Set[str] = None,
) -> str:
    """
    Parse Python code and return JSON representation

    Args:
        code: Python source code as string
        parser: tree-sitter parser instance
        module_name: Name of the module
        local_imports: List of local import paths to track (e.g., ['src.prompts.report_planner_instructions', 'src.agent.NutanixLLM'])
        dotted_folder_name: The dotted path of the current file's folder
        requirements_packages: Set of installed package names for categorization

    Returns:
        JSON string representation of the AST
    """
    tree = parser.parse(bytes(code, encoding="utf-8"))
    ast_parser = ASTToJSONParser(
        tree, code, local_imports, dotted_folder_name, requirements_packages
    )
    result = ast_parser.parse(module_name)
    print("1.###########", local_imports)
    print("2.###########", ast_parser.imported_names)
    return json.dumps(result, indent=2)


class CodeGraphBuilder:
    def __init__(self, url: str, username: str, password: str):
        """Initialize the Neo4j graph connection."""
        self.graph = Neo4jGraph(url=url, username=username, password=password)

    def create_constraints(self):
        """Create unique constraints for different node types."""
        constraints = [
            # Unique constraints for different node types
            "CREATE CONSTRAINT module_name_unique IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT class_name_signature_module_unique IF NOT EXISTS FOR (c:Class) REQUIRE (c.name, c.signature, c.module_name) IS UNIQUE",
            "CREATE CONSTRAINT function_name_signature_module_unique IF NOT EXISTS FOR (f:Function) REQUIRE (f.name, f.signature, f.module_name) IS UNIQUE",
            "CREATE CONSTRAINT method_name_class_signature_module_unique IF NOT EXISTS FOR (m:Method) REQUIRE (m.name, m.class, m.signature, m.module_name) IS UNIQUE",
            "CREATE CONSTRAINT field_name_class_module_unique IF NOT EXISTS FOR (f:Field) REQUIRE (f.name, f.class, f.module_name) IS UNIQUE",
            "CREATE CONSTRAINT global_var_name_module_unique IF NOT EXISTS FOR (g:GlobalVariable) REQUIRE (g.name, g.module_name) IS UNIQUE",
        ]

        indexes = [
            # Indexes for better performance
            "CREATE INDEX module_name_index IF NOT EXISTS FOR (m:Module) ON (m.name)",
            "CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX class_module_name_index IF NOT EXISTS FOR (c:Class) ON (c.module_name)",
            "CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name)",
            "CREATE INDEX function_module_name_index IF NOT EXISTS FOR (f:Function) ON (f.module_name)",
            "CREATE INDEX method_name_index IF NOT EXISTS FOR (m:Method) ON (m.name)",
            "CREATE INDEX method_module_name_index IF NOT EXISTS FOR (m:Method) ON (m.module_name)",
            "CREATE INDEX method_class_index IF NOT EXISTS FOR (m:Method) ON (m.class)",
            "CREATE INDEX field_name_index IF NOT EXISTS FOR (f:Field) ON (f.name)",
            "CREATE INDEX field_module_name_index IF NOT EXISTS FOR (f:Field) ON (f.module_name)",
            "CREATE INDEX field_class_index IF NOT EXISTS FOR (f:Field) ON (f.class)",
            "CREATE INDEX global_var_name_index IF NOT EXISTS FOR (g:GlobalVariable) ON (g.name)",
            "CREATE INDEX global_var_module_name_index IF NOT EXISTS FOR (g:GlobalVariable) ON (g.module_name)",
            "CREATE INDEX import_name_index IF NOT EXISTS FOR (i:Import) ON (i.name)",
            "CREATE INDEX import_module_index IF NOT EXISTS FOR (i:Import) ON (i.module)",
        ]

        all_statements = constraints + indexes

        for statement in all_statements:
            try:
                self.graph.query(statement)
                statement_type = "CONSTRAINT" if "CONSTRAINT" in statement else "INDEX"
                print(
                    f"✓ Applied {statement_type}: {statement.split(' ')[2] if statement_type == 'CONSTRAINT' else statement.split(' ')[2]}"
                )
            except Exception as e:
                print(f"✗ Failed to apply statement: {e}")

    def create_node(self, node_data: Dict[str, Any], repo_name: str) -> str:
        """Create a node based on its type and return the node identifier."""
        node_type = node_data.get("type")
        node_name = node_data.get("name")

        if node_type == "MODULE":
            local_name = node_data.get("local_name")
            query = """
            MERGE (m:Module {name: $name, local_name: $local_name})
            RETURN m
            """
            self.graph.query(query, {"name": node_name, "local_name": local_name})
            # Connect module to repo node if repo_name is provided
            if repo_name:
                connect_query = """
                MATCH (r:Repo {name: $repo_name})
                MATCH (m:Module {name: $module_name})
                MERGE (r)-[:CONTAINS]->(m)
                """
                self.graph.query(
                    connect_query, {"repo_name": repo_name, "module_name": node_name}
                )
                print(f"✓ Connected Module {node_name} to Repo {repo_name}")

            return f"Module:{node_name}"

        elif node_type == "Class":
            signature = node_data.get("signature", "")
            code = node_data.get("code", "")
            module_name = node_data.get("module_name", "")

            query = """
            MERGE (c:Class {name: $name, signature: $signature, module_name: $module_name})
            SET c.code = $code
            RETURN c
            """
            self.graph.query(
                query,
                {
                    "name": node_name,
                    "signature": signature,
                    "code": code,
                    "module_name": module_name,
                },
            )
            return f"Class:{module_name}:{node_name}"

        elif node_type == "Function":
            signature = node_data.get("signature", "")
            code = node_data.get("code", "")
            module_name = node_data.get("module_name", "")

            query = """
            MERGE (f:Function {name: $name, signature: $signature, module_name: $module_name})
            SET f.code = $code
            RETURN f
            """
            self.graph.query(
                query,
                {
                    "name": node_name,
                    "signature": signature,
                    "code": code,
                    "module_name": module_name,
                },
            )
            return f"Function:{module_name}:{node_name}"

        elif node_type == "Method":
            class_name = node_data.get("class")
            signature = node_data.get("signature", "")
            code = node_data.get("code", "")
            # Methods don't have module_name directly, but we can derive it from the class
            # For now, we'll add it as an optional field that can be populated separately
            module_name = node_data.get("module_name", "")

            query = """
            MERGE (m:Method {name: $name, class: $class, signature: $signature, module_name: $module_name})
            SET m.code = $code
            RETURN m
            """
            self.graph.query(
                query,
                {
                    "name": node_name,
                    "class": class_name,
                    "signature": signature,
                    "code": code,
                    "module_name": module_name,
                },
            )
            return f"Method:{class_name}:{node_name}"

        elif node_type == "Field":
            class_name = node_data.get("class")
            module_name = node_data.get("module_name", "")

            query = """
            MERGE (f:Field {name: $name, class: $class, module_name: $module_name})
            RETURN f
            """
            self.graph.query(
                query,
                {"name": node_name, "class": class_name, "module_name": module_name},
            )
            return f"Field:{class_name}:{node_name}"

        elif node_type == "GlobalVariable":
            code = node_data.get("code", "")
            module_name = node_data.get("module_name", "")

            query = """
            MERGE (g:GlobalVariable {name: $name, module_name: $module_name})
            SET g.code = $code
            RETURN g
            """
            self.graph.query(
                query, {"name": node_name, "code": code, "module_name": module_name}
            )
            return f"GlobalVariable:{module_name}:{node_name}"

        elif node_type == "IMPORT":
            module = node_data.get("module")
            alias = node_data.get("alias", "")
            dotted_folder_name = node_data.get("dotted_folder_name", "")

            query = """
            MERGE (i:Import {name: $name, module: $module})
            SET i.alias = $alias, i.dotted_folder_name = $dotted_folder_name
            RETURN i
            """
            self.graph.query(
                query,
                {
                    "name": node_name,
                    "module": module,
                    "alias": alias,
                    "dotted_folder_name": dotted_folder_name,
                },
            )
            return f"Import:{module}:{node_name}"

        return None

    def create_relationship(self, edge_data: Dict[str, Any], node_map: Dict[str, str]):
        """Create relationships between nodes."""
        rel_type = edge_data.get("type")
        source = edge_data.get("source")
        target = edge_data.get("target")
        source_assoc_type = edge_data.get("source_association_type")
        target_assoc_type = edge_data.get("target_association_type")

        # Create the relationship based on type
        if rel_type == "CONTAINS":
            self._create_contains_relationship(source, target, edge_data)
        elif rel_type == "HAS_METHOD":
            self._create_has_method_relationship(source, target, edge_data)
        elif rel_type == "HAS_FIELD":
            self._create_has_field_relationship(source, target, edge_data)
        elif rel_type == "INHERITS":
            self._create_inherits_relationship(source, target, edge_data)
        elif rel_type == "USES":
            self._create_uses_relationship(
                source, target, source_assoc_type, target_assoc_type, edge_data
            )

    def _create_contains_relationship(
        self, source_name: str, target_name: str, edge_data: Dict[str, Any]
    ):
        """Create CONTAINS relationship from MODULE to Class/Function/GlobalVariable/Import."""
        # Extract module information
        source_module = edge_data.get("source_module_name", source_name)
        target_module = edge_data.get("target_module_name")

        # Try Class first - match by name, signature, and module_name
        target_signature = edge_data.get("target_signature")
        if target_signature:
            query_class = """
            MATCH (m:Module {name: $source_name}), 
                (c:Class {name: $target_name, signature: $target_signature, module_name: $target_module})
            MERGE (m)-[:CONTAINS]->(c)
            RETURN count(*) as created
            """
            result = self.graph.query(
                query_class,
                {
                    "source_name": source_name,
                    "target_name": target_name,
                    "target_signature": target_signature,
                    "target_module": target_module,
                },
            )
            if result and result[0]["created"] > 0:
                return

        # Try Function - match by name, signature, and module_name
        if target_signature:
            query_function = """
            MATCH (m:Module {name: $source_name}), 
                (f:Function {name: $target_name, signature: $target_signature, module_name: $target_module})
            MERGE (m)-[:CONTAINS]->(f)
            RETURN count(*) as created
            """
            result = self.graph.query(
                query_function,
                {
                    "source_name": source_name,
                    "target_name": target_name,
                    "target_signature": target_signature,
                    "target_module": target_module,
                },
            )
            if result and result[0]["created"] > 0:
                return

        # Try GlobalVariable - match by name and module_name
        query_global = """
        MATCH (m:Module {name: $source_name}), 
            (g:GlobalVariable {name: $target_name, module_name: $target_module})
        MERGE (m)-[:CONTAINS]->(g)
        RETURN count(*) as created
        """
        self.graph.query(
            query_global,
            {
                "source_name": source_name,
                "target_name": target_name,
                "target_module": target_module,
            },
        )

        # Try Import node - match by name first, then by alias
        target_assoc_type = edge_data.get("target_association_type")
        if target_assoc_type == "IMPORT":
            # First try matching by name
            query_name = """
            MATCH (m:Module {name: $source_name}), 
                (i:Import {name: $target_name})
            MERGE (m)-[:CONTAINS]->(i)
            RETURN count(*) as created
            """
            params = {
                "source_name": source_name,
                "target_name": target_name,
            }
            result = self.graph.query(query_name, params)

            # If no match found by name, try matching by alias
            if not result or result[0]["created"] == 0:
                query_alias = """
                MATCH (m:Module {name: $source_name}), 
                    (i:Import {alias: $target_name})
                MERGE (m)-[:CONTAINS]->(i)
                RETURN count(*) as created
                """
                result = self.graph.query(query_alias, params)
            return

    def _create_has_method_relationship(
        self, class_name: str, method_name: str, edge_data: Dict[str, Any]
    ):
        """Create HAS_METHOD relationship from Class to Method."""
        # Extract required properties for unique identification
        class_signature = edge_data.get("source_signature")
        class_module = edge_data.get("source_module_name")
        method_signature = edge_data.get("target_signature")
        method_module = edge_data.get("target_module_name")
        method_class = edge_data.get("target_class", class_name)

        query = """
        MATCH (c:Class {name: $class_name, signature: $class_signature, module_name: $class_module}), 
            (m:Method {name: $method_name, class: $method_class, signature: $method_signature, module_name: $method_module})
        MERGE (c)-[:HAS_METHOD]->(m)
        """
        self.graph.query(
            query,
            {
                "class_name": class_name,
                "class_signature": class_signature,
                "class_module": class_module,
                "method_name": method_name,
                "method_class": method_class,
                "method_signature": method_signature,
                "method_module": method_module,
            },
        )

    def _create_has_field_relationship(
        self, class_name: str, field_name: str, edge_data: Dict[str, Any]
    ):
        """Create HAS_FIELD relationship from Class to Field."""
        # Extract required properties for unique identification
        class_signature = edge_data.get("source_signature")
        class_module = edge_data.get("source_module_name")
        field_module = edge_data.get("target_module_name")
        field_class = edge_data.get("target_class", class_name)

        query = """
        MATCH (c:Class {name: $class_name, signature: $class_signature, module_name: $class_module}), 
            (f:Field {name: $field_name, class: $field_class, module_name: $field_module})
        MERGE (c)-[:HAS_FIELD]->(f)
        """
        self.graph.query(
            query,
            {
                "class_name": class_name,
                "class_signature": class_signature,
                "class_module": class_module,
                "field_name": field_name,
                "field_class": field_class,
                "field_module": field_module,
            },
        )

    def _create_inherits_relationship(
        self, child_class: str, parent_class: str, edge_data: Dict[str, Any]
    ):
        """Create INHERITS relationship from Class to Class or Import node."""
        # Extract required properties for unique identification
        child_signature = edge_data.get("source_signature")
        child_module = edge_data.get("source_module_name")
        target_assoc_type = edge_data.get("target_association_type")

        # Check if target is an Import node
        if target_assoc_type == "IMPORT":
            # Create INHERITS relationship to Import node
            query_name = """
            MATCH (child:Class {name: $child_class, signature: $child_signature, module_name: $child_module})
            MATCH (parent:Import {name: $parent_class})
            MERGE (child)-[:INHERITS]->(parent)
            RETURN count(*) as created
            """
            params = {
                "child_class": child_class,
                "child_signature": child_signature,
                "child_module": child_module,
                "parent_class": parent_class,
            }
            result = self.graph.query(query_name, params)

            # If no match found by name, try matching by alias
            if not result or result[0]["created"] == 0:
                query_alias = """
                MATCH (child:Class {name: $child_class, signature: $child_signature, module_name: $child_module})
                MATCH (parent:Import {alias: $parent_class})
                MERGE (child)-[:INHERITS]->(parent)
                RETURN count(*) as created
                """
                result = self.graph.query(query_alias, params)

            created_count = result[0]["created"] if result else 0
            if created_count > 0:
                print(f"✓ Created INHERITS to Import: {child_class} -> {parent_class}")
            else:
                print(
                    f"⚠ Could not create INHERITS to Import: {child_class} -> {parent_class} (Import node not found)"
                )
        else:
            # Create INHERITS relationship to Class node (existing logic)
            parent_signature = edge_data.get("target_signature")
            parent_module = edge_data.get("target_module_name")

            query = """
            MATCH (child:Class {name: $child_class, signature: $child_signature, module_name: $child_module}), 
                (parent:Class {name: $parent_class, signature: $parent_signature, module_name: $parent_module})
            MERGE (child)-[:INHERITS]->(parent)
            """
            self.graph.query(
                query,
                {
                    "child_class": child_class,
                    "child_signature": child_signature,
                    "child_module": child_module,
                    "parent_class": parent_class,
                    "parent_signature": parent_signature,
                    "parent_module": parent_module,
                },
            )
            print(f"✓ Created INHERITS to Class: {child_class} -> {parent_class}")

    def _create_uses_relationship(
        self,
        source: str,
        target: str,
        source_type: str,
        target_type: str,
        edge_data: Dict[str, Any],
    ):
        """Create USES relationship from Function/Method to GlobalVariable/Field/Class/Function/IMPORT."""

        # Extract source properties with proper validation
        source_signature = edge_data.get("source_signature")
        source_module = edge_data.get("source_module_name")
        source_class = edge_data.get("source_class")

        # Extract target properties with proper validation
        target_signature = edge_data.get("target_signature")
        target_module = edge_data.get("target_module_name")
        target_class = edge_data.get("target_class")

        # Validate required fields are present
        if not source_module:
            print(f"✗ Missing source_module_name for {source_type} '{source}'")
            return

        if not target_module and target_type not in ["IMPORT"]:
            print(f"✗ Missing target_module_name for {target_type} '{target}'")
            return

        # Determine source node matching pattern
        if source_type == "Function":
            if not source_signature:
                print(f"✗ Missing source_signature for Function '{source}'")
                return
            source_match = "{name: $source, signature: $source_signature, module_name: $source_module}"
            source_label = "Function"
            source_params = {
                "source": source,
                "source_signature": source_signature,
                "source_module": source_module,
            }
        elif source_type == "Class":
            if not source_signature:
                print(f"✗ Missing source_signature for Class '{source}'")
                return
            source_match = "{name: $source, signature: $source_signature, module_name: $source_module}"
            source_label = "Class"
            source_params = {
                "source": source,
                "source_signature": source_signature,
                "source_module": source_module,
            }
        elif source_type == "Method":
            if not source_signature:
                print(f"✗ Missing source_signature for Method '{source}'")
                return
            if not source_class:
                print(f"✗ Missing source_class for Method '{source}'")
                return
            source_match = "{name: $source, class: $source_class, signature: $source_signature, module_name: $source_module}"
            source_label = "Method"
            source_params = {
                "source": source,
                "source_class": source_class,
                "source_signature": source_signature,
                "source_module": source_module,
            }
        elif source_type == "GlobalVariable":
            source_match = "{name: $source, module_name: $source_module}"
            source_label = "GlobalVariable"
            source_params = {
                "source": source,
                "source_module": source_module,
            }
        else:
            print(f"✗ Unknown source type: {source_type}")
            return

        # Create relationship based on target type
        if target_type == "GlobalVariable":
            target_match = "{name: $target, module_name: $target_module}"
            target_params = {"target": target, "target_module": target_module}

        elif target_type == "Field":
            if not target_class:
                print(f"✗ Missing target_class for Field '{target}'")
                return
            target_match = (
                "{name: $target, class: $target_class, module_name: $target_module}"
            )
            target_params = {
                "target": target,
                "target_class": target_class,
                "target_module": target_module,
            }

        elif target_type == "Class":
            if not target_signature:
                print(f"✗ Missing target_signature for Class '{target}'")
                return
            target_match = "{name: $target, signature: $target_signature, module_name: $target_module}"
            target_params = {
                "target": target,
                "target_signature": target_signature,
                "target_module": target_module,
            }

        elif target_type == "Function":
            if not target_signature:
                print(f"✗ Missing target_signature for Function '{target}'")
                return
            target_match = "{name: $target, signature: $target_signature, module_name: $target_module}"
            target_params = {
                "target": target,
                "target_signature": target_signature,
                "target_module": target_module,
            }

        elif target_type == "IMPORT":
            # For imports, try matching by name first
            query_name = f"""
            MATCH (s:{source_label} {source_match}), (t:Import {{name: $target}})
            MERGE (s)-[:USES]->(t)
            RETURN count(*) as created
            """
            params = {**source_params, "target": target}
            result = self.graph.query(query_name, params)

            # If no match found by name, try matching by alias
            if not result or result[0]["created"] == 0:
                query_alias = f"""
                MATCH (s:{source_label} {source_match}), (t:Import {{alias: $target}})
                MERGE (s)-[:USES]->(t)
                RETURN count(*) as created
                """
                self.graph.query(query_alias, params)
            return

        else:
            print(f"✗ Unknown target type: {target_type}")
            return

        # Execute the query for non-IMPORT targets
        query = f"""
        MATCH (s:{source_label} {source_match}), (t:{target_type} {target_match})
        MERGE (s)-[:USES]->(t)
        RETURN count(*) as created
        """

        all_params = {**source_params, **target_params}
        result = self.graph.query(query, all_params)

        # Log the result for debugging
        created_count = result[0]["created"] if result else 0
        if created_count > 0:
            print(
                f"✓ Created USES: {source_type} '{source}' -> {target_type} '{target}'"
            )
        else:
            print(
                f"⚠ No relationship created: {source_type} '{source}' -> {target_type} '{target}' (nodes may not exist)"
            )

    def build_graph_from_json(self, json_data: Dict[str, Any], repo_name: str):
        """Build the complete graph from JSON data."""
        print(" Creating constraints and indexes...")
        self.create_constraints()

        print("\n Creating nodes")
        node_map = {}

        # Create all nodes first
        for node in json_data.get("nodes", []):
            node_id = self.create_node(node, repo_name)
            if node_id:
                node_map[node_id] = node
                print(f"✓ Created: {node_id}")

        print(f"\n Creating relationships")
        # Create relationships
        for edge in json_data.get("edges", []):
            try:
                self.create_relationship(edge, node_map)
                rel_type = edge.get("type")
                source = edge.get("source")
                target = edge.get("target")
                print(f"✓ Created: {source} -{rel_type}-> {target}")
            except Exception as e:
                print(
                    f"✗ Failed to create relationship: {edge.get('source')} -> {edge.get('target')}: {e}"
                )

        print("\n Graph creation completed!")

    def clear_graph(self):
        """Clear the entire graph (useful for testing)."""
        self.graph.query("MATCH (n) DETACH DELETE (n)")
        print(" Graph cleared!")

    def create_repo_node(self, repo_name: str = "Repository"):
        """Create only the Repo node without connecting to modules."""
        print(f"Creating Repo node: {repo_name}")

        # Create the Repo node
        create_repo_query = """
        MERGE (r:Repo {name: $repo_name})
        RETURN r
        """
        self.graph.query(create_repo_query, {"repo_name": repo_name})
        print(f"Created Repo node: {repo_name}")

        return f"Repo:{repo_name}"

    def delete_field_nodes(self):
        """Delete all Field nodes and their relationships."""
        print("Deleting all Field nodes...")

        # Count Field nodes before deletion
        count_query = """
        MATCH (f:Field)
        RETURN count(f) as field_count
        """
        result = self.graph.query(count_query)
        field_count = result[0]["field_count"] if result else 0

        if field_count == 0:
            print("No Field nodes found to delete.")
            return

        # Delete all Field nodes and their relationships
        delete_query = """
        MATCH (f:Field)
        DETACH DELETE f
        RETURN count(*) as deleted_count
        """
        self.graph.query(delete_query)
        print(f"✓ Deleted {field_count} Field nodes and their relationships")


def merge_import_nodes(url, username, password):
    """
    Merge Import nodes with Class/Function/GlobalVariable nodes that have the same name.
    Redirects all USES relationships from Import nodes to corresponding target nodes.

    Args:
        url: Neo4j database URL
        username: Neo4j username
        password: Neo4j password
    """
    # Initialize Neo4j connection
    graph = Neo4jGraph(url=url, username=username, password=password)

    try:
        # Step 1: Find all Import nodes and extract their components
        find_imports_query = """
        MATCH (import:Import)
        RETURN import.name as import_name,
               import.module as import_module,
               import.dotted_folder_name as dotted_folder_name,
               elementId(import) as import_id
        """

        imports = graph.query(find_imports_query)

        if not imports:
            print("No Import nodes found.")
            return

        print(f"Found {len(imports)} Import nodes to process")

        total_merged = 0

        for import_node in imports:
            import_name = import_node["import_name"]
            import_module = import_node["import_module"]
            dotted_folder_name = import_node["dotted_folder_name"]
            import_id = import_node["import_id"]

            print(
                f"\nProcessing Import: name='{import_name}', module='{import_module}', folder_name='{dotted_folder_name}'"
            )

            # Stage 0: Check if import_module directly matches a Module node's name or local_name
            print(
                f"  Stage 0: Checking if import module '{import_module}' matches any Module node"
            )

            direct_module_match = attempt_direct_module_merge(
                graph,
                import_name,
                import_module,
                import_id,
                stage=0,
            )

            if direct_module_match:
                total_merged += 1
                continue

            # Extract module_name from import_module (everything before the last dot)
            if "." in import_module:
                module_name = ".".join(import_module.split(".")[:-1])
                target_name = import_module.split(".")[-1]
            else:
                module_name = import_module
                target_name = import_name

            print(
                f"  Extracted: module_name='{module_name}', target_name='{target_name}'"
            )

            # Stage 1: Try dotted_folder_name + '.' + module_name (only if dotted_folder_name exists)
            if dotted_folder_name:
                stage1_module_name = f"{dotted_folder_name}.{module_name}"
                print(f"  Stage 1: Looking for module '{stage1_module_name}'")

                merge_result = attempt_merge(
                    graph,
                    import_name,
                    import_module,
                    import_id,
                    stage1_module_name,
                    target_name,
                    stage=1,
                )

                if merge_result:
                    total_merged += 1
                    continue

            # Stage 2: Try just module_name
            print(f"  Stage 2: Looking for module '{module_name}'")

            merge_result = attempt_merge(
                graph,
                import_name,
                import_module,
                import_id,
                module_name,
                target_name,
                stage=2,
            )

            if merge_result:
                total_merged += 1
                continue

            # Stage 3: Name-based fallback matching (NEW!)
            print(
                f"  Stage 3: Attempting name-based fallback matching for '{import_name}'"
            )

            name_based_match = attempt_name_based_merge(
                graph, import_name, import_module, import_id, stage=3
            )

            if name_based_match:
                total_merged += 1
            else:
                print(f"   No matching target found for Import '{import_name}'")

        print(f"\n Successfully merged {total_merged} Import nodes")

    except Exception as e:
        print(f" Error during merge operation: {str(e)}")
        raise
    graph.close()


def attempt_merge(
    graph, import_name, import_module, import_id, module_name, target_name, stage
):
    """
    Attempt to merge an Import node with a target node (Class/Function/GlobalVariable).

    Returns True if merge was successful, False otherwise.
    """

    # Query to find matching module and target nodes
    find_target_query = """
    MATCH (module:Module {name: $module_name})
    OPTIONAL MATCH (class:Class {name: $target_name})<-[:CONTAINS]-(module)
    OPTIONAL MATCH (function:Function {name: $target_name})<-[:CONTAINS]-(module)
    OPTIONAL MATCH (globalvar:GlobalVariable {name: $target_name})<-[:CONTAINS]-(module)
    RETURN module.name as module_name,
           class.name as class_name,
           function.name as function_name,
           globalvar.name as globalvar_name,
           elementId(class) as class_id,
           elementId(function) as function_id,
           elementId(globalvar) as globalvar_id
    """

    targets = graph.query(
        find_target_query,
        params={"module_name": module_name, "target_name": target_name},
    )

    if not targets or not targets[0]["module_name"]:
        print(f"    No module '{module_name}' found")
        return False

    target = targets[0]
    target_node = None
    target_type = None
    target_element_id = None

    # Determine which type of target node was found
    if target["class_name"]:
        target_node = "Class"
        target_type = "Class"
        target_element_id = target["class_id"]
        print(f"    Found Class '{target['class_name']}' in module '{module_name}'")
    elif target["function_name"]:
        target_node = "Function"
        target_type = "Function"
        target_element_id = target["function_id"]
        print(
            f"    Found Function '{target['function_name']}' in module '{module_name}'"
        )
    elif target["globalvar_name"]:
        target_node = "GlobalVariable"
        target_type = "GlobalVariable"
        target_element_id = target["globalvar_id"]
        print(
            f"    Found GlobalVariable '{target['globalvar_name']}' in module '{module_name}'"
        )
    else:
        print(
            f"    No matching Class/Function/GlobalVariable '{target_name}' found in module '{module_name}'"
        )
        return False

    # Perform the merge
    merge_query = f"""
    MATCH (import:Import {{name: $import_name, module: $import_module}})
    WHERE elementId(import) = $import_id
    MATCH (target:{target_node} {{name: $target_name}})
    WHERE elementId(target) = $target_element_id
    
    // Find all USES relationships pointing to the Import node
    OPTIONAL MATCH (source)-[r:USES]->(import)
    
    // Create new USES relationships to target node (if any exist)
    FOREACH (rel IN CASE WHEN r IS NOT NULL THEN [r] ELSE [] END |
        MERGE (source)-[uses:USES]->(target)
        SET uses.source_association_type = CASE WHEN rel.source_association_type IS NOT NULL 
                                           THEN rel.source_association_type 
                                           ELSE "UNKNOWN" END,
            uses.target_association_type = $target_type
    )
    
    // Just count relationships (don't delete anything yet)
    RETURN count(r) as relationship_count
    """

    result = graph.query(
        merge_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
            "target_name": target_name,
            "target_element_id": target_element_id,
            "target_type": target_type,
        },
    )

    relationships_count = result[0]["relationship_count"] if result else 0

    # Handle INHERITS relationships separately (simpler and cleaner)
    inherits_query = """
    MATCH (source)-[r:INHERITS]->(import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    RETURN elementId(source) as source_id, source.name as source_name
    """

    inherits_rels = graph.query(
        inherits_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    inherits_count = 0
    for rel in inherits_rels:
        # Create new INHERITS relationship to target Class
        create_inherits_query = f"""
        MATCH (source) WHERE elementId(source) = $source_id
        MATCH (target:{target_node} {{name: $target_name}}) WHERE elementId(target) = $target_element_id
        MERGE (source)-[:INHERITS]->(target)
        """

        graph.query(
            create_inherits_query,
            params={
                "source_id": rel["source_id"],
                "target_name": target_name,
                "target_element_id": target_element_id,
            },
        )
        inherits_count += 1
        print(f"    ✓ Redirected INHERITS: {rel['source_name']} -> {target_name}")

    # Handle CONTAINS relationships separately (for __init__.py imports)
    contains_query = """
    MATCH (source)-[r:CONTAINS]->(import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    RETURN elementId(source) as source_id, source.name as source_name
    """

    contains_rels = graph.query(
        contains_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    contains_count = 0
    for rel in contains_rels:
        # Create new CONTAINS relationship to target node
        create_contains_query = f"""
        MATCH (source) WHERE elementId(source) = $source_id
        MATCH (target:{target_node} {{name: $target_name}}) WHERE elementId(target) = $target_element_id
        MERGE (source)-[:CONTAINS]->(target)
        """

        graph.query(
            create_contains_query,
            params={
                "source_id": rel["source_id"],
                "target_name": target_name,
                "target_element_id": target_element_id,
            },
        )
        contains_count += 1
        print(f"    ✓ Redirected CONTAINS: {rel['source_name']} -> {target_name}")

    total_relationships = relationships_count + inherits_count + contains_count
    print(
        f"  Stage {stage}: Merged Import '{import_name}' -> {target_node} '{target_name}' ({relationships_count} USES + {inherits_count} INHERITS + {contains_count} CONTAINS = {total_relationships} total relationships redirected)"
    )

    # DETACH DELETE removes the Import node AND ALL its relationships in one go
    detach_delete_query = """
    MATCH (import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    DETACH DELETE import
    """
    graph.query(
        detach_delete_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    return True


def attempt_direct_module_merge(graph, import_name, import_module, import_id, stage):
    """
    Attempt to merge an Import node directly with a Module node if the import's module
    attribute matches the Module's name or local_name.

    Returns True if merge was successful, False otherwise.
    """

    # Query to find Module nodes that match the import_module by name or local_name
    find_module_query = """
    MATCH (module:Module)
    WHERE module.name = $import_module OR module.local_name = $import_module
    RETURN module.name as module_name, 
           module.local_name as module_local_name,
           elementId(module) as module_id
    """

    modules = graph.query(find_module_query, params={"import_module": import_module})

    if not modules:
        print(f"    No Module found with name or local_name matching '{import_module}'")
        return False

    module = modules[0]
    module_name = module["module_name"]
    module_local_name = module["module_local_name"]
    module_element_id = module["module_id"]

    print(
        f"    Found Module '{module_name}' (local_name: '{module_local_name}') matching import module '{import_module}'"
    )

    # Perform the merge by redirecting ALL relationships from Import to Module
    merge_query = """
    MATCH (import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    MATCH (target_module:Module)
    WHERE elementId(target_module) = $module_element_id
    
    // Find all USES relationships pointing to the Import node
    OPTIONAL MATCH (source)-[r:USES]->(import)
    
    // Create new USES relationships to Module node (if any exist)
    FOREACH (rel IN CASE WHEN r IS NOT NULL THEN [r] ELSE [] END |
        MERGE (source)-[uses:USES]->(target_module)
        SET uses.source_association_type = CASE WHEN rel.source_association_type IS NOT NULL 
                                           THEN rel.source_association_type 
                                           ELSE "UNKNOWN" END,
            uses.target_association_type = "MODULE"
    )
    
    // Just count USES relationships (don't delete anything yet)
    RETURN count(r) as relationship_count
    """

    result = graph.query(
        merge_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
            "module_element_id": module_element_id,
        },
    )

    relationships_count = result[0]["relationship_count"] if result else 0

    # Handle INHERITS relationships
    inherits_query = """
    MATCH (source)-[r:INHERITS]->(import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    RETURN elementId(source) as source_id, source.name as source_name
    """

    inherits_rels = graph.query(
        inherits_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    inherits_count = 0
    for rel in inherits_rels:
        # Create new INHERITS relationship to target Module
        create_inherits_query = """
        MATCH (source) WHERE elementId(source) = $source_id
        MATCH (target_module:Module) WHERE elementId(target_module) = $module_element_id
        MERGE (source)-[:INHERITS]->(target_module)
        """

        graph.query(
            create_inherits_query,
            params={
                "source_id": rel["source_id"],
                "module_element_id": module_element_id,
            },
        )
        inherits_count += 1
        print(f"    ✓ Redirected INHERITS: {rel['source_name']} -> {module_name}")

    # Handle CONTAINS relationships (for __init__.py imports)
    contains_query = """
    MATCH (source)-[r:CONTAINS]->(import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    RETURN elementId(source) as source_id, source.name as source_name
    """

    contains_rels = graph.query(
        contains_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    contains_count = 0
    for rel in contains_rels:
        # Create new CONTAINS relationship to target Module
        create_contains_query = """
        MATCH (source) WHERE elementId(source) = $source_id
        MATCH (target_module:Module) WHERE elementId(target_module) = $module_element_id
        MERGE (source)-[:CONTAINS]->(target_module)
        """

        graph.query(
            create_contains_query,
            params={
                "source_id": rel["source_id"],
                "module_element_id": module_element_id,
            },
        )
        contains_count += 1
        print(f"    ✓ Redirected CONTAINS: {rel['source_name']} -> {module_name}")

    total_relationships = relationships_count + inherits_count + contains_count
    print(
        f"  Stage {stage}: Merged Import '{import_name}' -> Module '{module_name}' ({relationships_count} USES + {inherits_count} INHERITS + {contains_count} CONTAINS = {total_relationships} total relationships redirected)"
    )

    # NOW it's safe to use DETACH DELETE after all relationships are handled
    detach_delete_query = """
    MATCH (import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    DETACH DELETE import
    """
    graph.query(
        detach_delete_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    return True


def remove_self_loops(url, username, password):
    """
    Remove all self-loop relationships (nodes connected to themselves).

    Args:
        url: Neo4j database URL
        username: Neo4j username
        password: Neo4j password
    """
    # Initialize Neo4j connection
    graph = Neo4jGraph(url=url, username=username, password=password)

    try:
        print("Removing self-loop relationships...")

        # Query to find and count self-loop relationships
        count_query = """
        MATCH (n)-[r]->(n)
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY relationship_type
        """

        self_loops = graph.query(count_query)

        if not self_loops:
            print("No self-loop relationships found.")
            return

        # Display what we found
        total_loops = 0
        print("Found self-loop relationships:")
        for loop in self_loops:
            rel_type = loop["relationship_type"]
            count = loop["count"]
            total_loops += count
            print(f"  {rel_type}: {count} relationships")

        print(f"Total self-loops found: {total_loops}")

        # Remove all self-loop relationships
        remove_query = """
        MATCH (n)-[r]->(n)
        DELETE r
        RETURN count(r) as deleted_count
        """

        result = graph.query(remove_query)
        deleted_count = result[0]["deleted_count"] if result else 0

        print(f"✓ Successfully removed {deleted_count} self-loop relationships")

    except Exception as e:
        print(f"✗ Error during self-loop removal: {str(e)}")
        raise
    finally:
        graph.close()


def attempt_name_based_merge(graph, import_name, import_module, import_id, stage):
    """
    Attempt to merge an Import node with any target node that has the same name,
    regardless of module structure. This is a fallback when module-based matching fails.

    Returns True if merge was successful, False otherwise.
    """

    print(f"    🔍 Searching for any node with name '{import_name}'")

    # Query to find any Class/Function/GlobalVariable with matching name
    find_any_target_query = """
    OPTIONAL MATCH (class:Class {name: $import_name})
    OPTIONAL MATCH (function:Function {name: $import_name})  
    OPTIONAL MATCH (globalvar:GlobalVariable {name: $import_name})
    WITH class, function, globalvar
    WHERE class IS NOT NULL OR function IS NOT NULL OR globalvar IS NOT NULL
    RETURN 
        class.name as class_name,
        class.module_name as class_module,
        elementId(class) as class_id,
        function.name as function_name,
        function.module_name as function_module,
        elementId(function) as function_id,
        globalvar.name as globalvar_name,
        globalvar.module_name as globalvar_module,
        elementId(globalvar) as globalvar_id
    """

    targets = graph.query(find_any_target_query, params={"import_name": import_name})

    if not targets or not any(targets[0].values()):
        print(f"    ❌ No nodes found with name '{import_name}'")
        return False

    target = targets[0]
    target_node = None
    target_type = None
    target_element_id = None
    target_module = None

    # Determine which type of target node was found (prioritize in this order)
    if target["class_name"]:
        target_node = "Class"
        target_type = "Class"
        target_element_id = target["class_id"]
        target_module = target["class_module"]
        print(
            f"    ✅ Found Class '{target['class_name']}' in module '{target_module}'"
        )
    elif target["globalvar_name"]:
        target_node = "GlobalVariable"
        target_type = "GlobalVariable"
        target_element_id = target["globalvar_id"]
        target_module = target["globalvar_module"]
        print(
            f"    ✅ Found GlobalVariable '{target['globalvar_name']}' in module '{target_module}'"
        )
    elif target["function_name"]:
        target_node = "Function"
        target_type = "Function"
        target_element_id = target["function_id"]
        target_module = target["function_module"]
        print(
            f"    ✅ Found Function '{target['function_name']}' in module '{target_module}'"
        )

    # Perform the merge
    merge_query = f"""
    MATCH (import:Import {{name: $import_name, module: $import_module}})
    WHERE elementId(import) = $import_id
    MATCH (target:{target_node} {{name: $import_name}})
    WHERE elementId(target) = $target_element_id
    
    // Find all USES relationships pointing to the Import node
    OPTIONAL MATCH (source)-[r:USES]->(import)
    
    // Create new USES relationships to target node (if any exist)
    FOREACH (rel IN CASE WHEN r IS NOT NULL THEN [r] ELSE [] END |
        MERGE (source)-[uses:USES]->(target)
        SET uses.source_association_type = CASE WHEN rel.source_association_type IS NOT NULL 
                                           THEN rel.source_association_type 
                                           ELSE "UNKNOWN" END,
            uses.target_association_type = $target_type
    )
    
    // Just count relationships (don't delete anything yet)
    RETURN count(r) as relationship_count
    """

    result = graph.query(
        merge_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
            "target_element_id": target_element_id,
            "target_type": target_type,
        },
    )

    relationships_count = result[0]["relationship_count"] if result else 0

    # Handle INHERITS relationships
    inherits_query = """
    MATCH (source)-[r:INHERITS]->(import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    RETURN elementId(source) as source_id, source.name as source_name
    """

    inherits_rels = graph.query(
        inherits_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    inherits_count = 0
    for rel in inherits_rels:
        # Create new INHERITS relationship to target Class
        create_inherits_query = f"""
        MATCH (source) WHERE elementId(source) = $source_id
        MATCH (target:{target_node} {{name: $import_name}}) WHERE elementId(target) = $target_element_id
        MERGE (source)-[:INHERITS]->(target)
        """

        graph.query(
            create_inherits_query,
            params={
                "source_id": rel["source_id"],
                "import_name": import_name,
                "target_element_id": target_element_id,
            },
        )
        inherits_count += 1
        print(f"    ✓ Redirected INHERITS: {rel['source_name']} -> {import_name}")

    # Handle CONTAINS relationships (for __init__.py imports)
    contains_query = """
    MATCH (source)-[r:CONTAINS]->(import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    RETURN elementId(source) as source_id, source.name as source_name
    """

    contains_rels = graph.query(
        contains_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    contains_count = 0
    for rel in contains_rels:
        # Create new CONTAINS relationship to target node
        create_contains_query = f"""
        MATCH (source) WHERE elementId(source) = $source_id
        MATCH (target:{target_node} {{name: $import_name}}) WHERE elementId(target) = $target_element_id
        MERGE (source)-[:CONTAINS]->(target)
        """

        graph.query(
            create_contains_query,
            params={
                "source_id": rel["source_id"],
                "import_name": import_name,
                "target_element_id": target_element_id,
            },
        )
        contains_count += 1
        print(f"    ✓ Redirected CONTAINS: {rel['source_name']} -> {import_name}")

    total_relationships = relationships_count + inherits_count + contains_count
    print(
        f"  ✅ Stage {stage}: Merged Import '{import_name}' -> {target_node} '{import_name}' in '{target_module}' ({relationships_count} USES + {inherits_count} INHERITS + {contains_count} CONTAINS = {total_relationships} total relationships redirected)"
    )

    # DETACH DELETE removes the Import node AND ALL its relationships in one go
    detach_delete_query = """
    MATCH (import:Import {name: $import_name, module: $import_module})
    WHERE elementId(import) = $import_id
    DETACH DELETE import
    """
    graph.query(
        detach_delete_query,
        params={
            "import_name": import_name,
            "import_module": import_module,
            "import_id": import_id,
        },
    )

    return True


def remove_remaining_import_nodes(url, username, password):
    """
    Remove all remaining Import nodes and their relationships.

    Args:
        url: Neo4j database URL
        username: Neo4j username
        password: Neo4j password
    """
    from langchain_neo4j import Neo4jGraph

    # Initialize Neo4j connection
    graph = Neo4jGraph(url=url, username=username, password=password)

    try:
        # Count Import nodes before deletion
        count_query = """
        MATCH (i:Import)
        RETURN count(i) as import_count
        """
        result = graph.query(count_query)
        import_count = result[0]["import_count"] if result else 0

        if import_count == 0:
            print("No Import nodes found to delete.")
            return

        print(f"Found {import_count} Import nodes to delete")

        # Delete all Import nodes and their relationships
        delete_query = """
        MATCH (i:Import)
        DETACH DELETE i
        RETURN count(*) as deleted_count
        """

        graph.query(delete_query)
        print(
            f"✓ Successfully deleted {import_count} Import nodes and their relationships"
        )

    except Exception as e:
        print(f"✗ Error during Import node deletion: {str(e)}")
        raise
    finally:
        graph.close()


def for_each_file(
    file_path, requirements_path, dotted_folder_path, url, username, password, repo_name
):
    try:
        with open(file_path, "r") as f:
            code = f.read()
        match = re.search(r"[^/\\]+$", file_path)

        if match:
            filename = match.group()
        requirements_packages = load_requirements(requirements_path=requirements_path)
        parser = get_parser("python")

        # Handle special case for __init__.py files
        base_filename = filename[:-3]  # Remove .py extension
        if base_filename == "__init__":
            # For __init__.py files, use just the folder name as module name
            if dotted_folder_path:
                module_name = dotted_folder_path
            else:
                # For __init__.py in root directory, use a default name
                module_name = "main"
        else:
            # For regular .py files, use folder.filename format
            if dotted_folder_path:
                module_name = dotted_folder_path + "." + base_filename
            else:
                # For files in root directory, use just the filename
                module_name = base_filename

        result_json = parse_code_to_json(
            code,
            parser,
            module_name,
            local_imports=None,  # No longer needed - categorization happens inside parser
            dotted_folder_name=dotted_folder_path,
            requirements_packages=requirements_packages,
        )

        json_data = json.loads(result_json)

        # Create graph builder and build the graph
        builder = CodeGraphBuilder(url, username, password)
        builder.build_graph_from_json(json_data, repo_name)
        builder.delete_field_nodes()
        builder.graph.close()
    except Exception as e:
        print(f"✗ Error during {file_path}: {e}")
        return


def main(root_dir, requirements_path, url, username, password):
    root_dir = os.path.abspath(root_dir)
    result = []
    builder = CodeGraphBuilder(url, username, password)
    builder.create_repo_node(root_dir)
    builder.graph.close()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get the relative path from root_dir
        rel_path = os.path.relpath(dirpath, root_dir)

        # Skip '.' and convert to dotted notation
        if rel_path == ".":
            dotted_path = ""
        else:
            dotted_path = rel_path.replace(os.sep, ".")

        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                print(f"Processing file: {full_path}")
                for_each_file(
                    full_path,
                    requirements_path,
                    dotted_path,
                    url,
                    username,
                    password,
                    root_dir,
                )

    merge_import_nodes(url, username, password)
    remove_self_loops(url, username, password)

    remove_remaining_import_nodes(url, username, password)
    builder = CodeGraphBuilder(url, username, password)
    builder.graph.close()

if __name__ == "__main__":
    root_directory = "./path/to/your/repository"  # Update with your repository path
    requirements_path = (
        ""
    )
    url = "bolt://localhost:7687"  # Replace with your Neo4j URL
    username = "neo4j"  # Replace with your username
    password = "your_neo4j_password"  # Replace with your Neo4j password
    main(root_directory, requirements_path, url, username, password)
