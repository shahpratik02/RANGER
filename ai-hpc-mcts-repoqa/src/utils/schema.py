nodes_description = """
1. **Module**:
   - **Attributes**:
     - `name` (String): Name of the module (dotted name)
     - `local_name` (String): The local name of the module (without the dotted path)
     - 'embedding': Embedding for the module using Code2Json


2. **Class**:
   - **Attributes**:
     - `name` (String): Name of the class
     - `signature` (String): The signature of the class
     - `code` (String): Full code of the class
     - 'module_name' (String): Name of the module the class belongs to
     - 'embedding': Embedding for the class using Code2Json

3. **Function**:
   - **Attributes**:
     - `name` (String): Name of the function
     - `code` (String): Full code of the function
     - `signature` (String): The signature of the function
     - `module_name` (String): Name of the module the function belongs to
     - 'embedding': Embedding for the function using Code2Json

4. **Field**:
   - **Attributes**:
     - `name` (String): Name of the field
     - `class` (String): Name of the class the field belongs to
   

5. **Method**:
   - **Attributes**:
     - `name` (String): Name of the method
     - `class` (String): Name of the class the method belongs to
     - `code` (String): Full code of the method
     - `signature` (String): The signature of the method
     - `module_name` (String): Name of the module the method belongs to
     - 'embedding': Embedding for the method using Code2Json

6. **GlobalVariable**:
   - **Attributes**:
     - `name` (String): Name of the global variable
     - `code` (String): The code segment in which the global variable is defined.
     - `module_name` (String): Name of the module the global variable belongs to
     - 'embedding': Embedding for the global variable using Code2Json

7. **Repo**:
   - **Attributes**:
     - `name` (String): Name of the repository
"""

edges_description = """
1. **CONTAINS**:
   - **Source**: MODULE or REPO
   - **Target**: MODULE or CLASS or FUNCTION or GLOBAL_VARIABLE

2. **HAS_METHOD**:
   - **Source**: CLASS
   - **Target**: METHOD

3. **HAS_FIELD**:
   - **Source**: CLASS
   - **Target**: FIELD

4. **INHERITS**:
   - **Source**: CLASS
   - **Target**: CLASS (base class)

5. **USES**:
   - **Source**: CLASS,FUNCTION, METHOD or GLOBAL_VARIABLE
   - **Target**: GLOBAL_VARIABLE, FIELD, CLASS, FUNCTION, or IMPORT
   
"""
# - **Attributes**:
#      - `source_association_type` (String): `FUNCTION`, `METHOD`, 'GLOBAL_VARIABLE'
#      - `target_association_type` (String): `GLOBAL_VARIABLE`, `FIELD`, `CLASS`, `FUNCTION`, `IMPORT`
# 7. **Import**:
#    - **Attributes**:
#      - `name` (String): Name of the imported item (module, class, function, or variable)
#      - `module` (String): Source module name
#      - `alias` (String, optional): Alias if imported with 'as'
#      - 'dotted_folder_name' (String, optional): Dotted path if imported from a submodule