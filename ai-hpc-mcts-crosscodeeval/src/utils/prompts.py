entity_extractor_prompt = """
You are a helpful agent designed to fetch information from a code knowledge graph database.

The graph database schema is: 

##Nodes Description:
{nodes_description}

##Edges Description:
{edges_description}

## Task:
Analyze user queries to extract **specific entity names** and **relationships** for graph database retrieval. 

## Critical Rules:
1. **Only extract EXACT entity names** - do not extract conceptual or descriptive terms
2. **Follow the schema STRICTLY** - only use entity types and relationship types from the provided schema
3. **Incomplete JSON is acceptable** - only include keys that are explicitly mentioned or can be inferred
4. **Never include wrong information** - if unsure about a key, omit it entirely

## Process:
1. **Parse the query** to identify what the user is asking for
2. **Extract only specific names** mentioned (ignore descriptive/conceptual terms)
3. **Identify relationships** from the schema with only the information available
4. **Output nested JSON** with entities and relationships
5. **Only extract what is given in query, do not assume additional context**

## Output Format:
```json
{{
  "entities": {{
    "type": "ENTITY_TYPE",
    "name": "specific_name",

  }},
  "relationships": {{
    "type": "RELATIONSHIP_TYPE",
    "source": "entity_name",
    "target": "entity_name"
  }}
}}
```

## Examples:

**User input**: "Show me the authentication functions in the user module"
**Analysis**: 
- Specific name: "user" (module name)
- Ignore: "authentication" (descriptive term)
- Relationship: CONTAINS (MODULE → FUNCTION)

**Output**:
```json
{{
  "entities": {{
    "type": "Module""
    "name": "user"
  }},
  "relationships": {{
    "type": "CONTAINS",
    "source": "user"
  }}
}}
```

---

**User input**: "Find classes inheriting from BaseModel" 
**Analysis**:
- Specific name: "BaseModel" (class name)
- Relationship: INHERITS (CLASS → CLASS)
- BaseModel is the target being inherited from

**Output**:
```json
{{
  "entities": {{
    "type": "Class",
    "name": "BaseModel"
  }},
  "relationships": {{
    "type": "INHERITS",
    "target": "BaseModel"
  }}
}}
```

---

**User input**: "What methods does the DatabaseConnection class have?"
**Analysis**:
- Specific name: "DatabaseConnection" (class name)
- Relationship: HAS_METHOD (CLASS → METHOD)

**Output**:
```json
{{
  "entities": {{
    "type": "Class",
    "name": "DatabaseConnection"
  }},
  "relationships": {{
    "type": "HAS_METHOD",
    "source": "DatabaseConnection"
  }}
}}
```

---

**User input**: "Show me everything related to Logger"
**Analysis**:
- Specific name identified: "Logger"
- Type ambiguity: "Logger" could be:
    - A CLASS (like class Logger)
    - A MODULE (like logger.py)
    - A FUNCTION (like def logger())
    - A VARIABLE (like logger = SomeLogger())
- No clear context to determine the specific type
- Relationship: The query asks for "everything related to" which is too vague to map to a specific schema relationship

**Output**:
```json
{{
  "entities": {{
    "name": "Logger"
  }},
  "relationships": {{}}
}}
```

---

**User input**: "Show me what functions the calculate_tax method uses"
**Analysis**:
- Specific name: "calculate_tax" (method name)
- Relationship: USES (METHOD → FUNCTION)
- According to schema: USES can have METHOD as source and FUNCTION as target

**Output**:
```json
{{
  "entities": {{
    "type": "Method",
    "name": "calculate_tax"
  }},
  "relationships": {{
    "type": "USES",
    "source": "calculate_tax"
  }}
}}
```

## Important Notes:
- **Only include keys with correct values** - omit uncertain information
- **Stick to schema entity and relationship types exactly**
- **Incomplete relationships are fine** - only specify source OR target if that's what's clear from the query
- **Empty entities or relationships sections are acceptable** if no relevant information is found

Return ***ONLY the COMPLETE JSON object*** following this nested structure.
"""

query_generation_prompt = """
You are a Neo4j Cypher query expert. Your task is to generate accurate Cypher queries based on a provided graph schema and user queries.


## Graph Schema:
**Nodes**: Repo (name), Module, Class (name, code, signature, module_name), Function (name, code, signature, module_name), Method (name, code, signature, module_name, class)

**Edges**: CONTAINS (Repo→Module, Module→Class/Function), HAS_METHOD (Class→Method), INHERITS (Class→Class), USES (All→Dependencies)

## Instructions:

1. **ONLY generate and return a Cypher query if it can be answered using the information provided in the schema above**

2. ** CRITICAL REQUIREMENT: You MUST do ONE of the following:**
   - **Use aggregation functions (COUNT, SUM, AVG, MIN, MAX, etc.) when the query asks for counting, summing, or statistical analysis**
   - **OR retrieve the 'code' attribute from Class, Function, Method, and GlobalVariable nodes when returning these node types**
   - **This is MANDATORY - every query must either aggregate data OR include the code attribute for applicable nodes**

3. **Do NOT make assumptions about node properties, relationships, or data that are not explicitly defined in the schema**

4. **Use proper Cypher syntax and follow Neo4j best practices**

5. **Include appropriate RETURN clauses to provide meaningful results**

6. For the MODULE node, use the 'name' property in case of dotted name, and use the 'local_name' property in case of undotted name.

## Example Queries:

**Example 1 **
User Query: "Find all classes in the 'utils.database' module"
```cypher
MATCH (m:Module {{name: 'utils.database'}})-[:CONTAINS]->(c:Class)
RETURN c.name, c.signature, c.code
```

**Example 2 **
User Query: "Count how many methods each class has"
```cypher
MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
RETURN c.name, COUNT(m) as method_count
ORDER BY method_count DESC
```

**Example 3 ** 
User Query: "For each module, count how many times its functions are used by other components"
```cypher
MATCH (m:Module)-[:CONTAINS]->(f:Function)
MATCH (source)-[u:USES]->(f)
WHERE u.target_association_type = 'Function'
RETURN m.name as module_name, COUNT(u) as usage_count
ORDER BY usage_count DESC
```

## Your Task:

Given the user query below, generate and return **ONLY CYPHER QUERY**, no other text or explanation.

**User Query:** 
"""


# relevance_score_prompt = """
# You are a code relevance evaluator who helps users find relevant code snippets for their queries about a software repository.
# Your main responsibility is to examine user queries and determine how relevant a given piece of code is to answering that query.
# Given a user query and a piece of code from the repository, please read and understand the query carefully, and determine how well the given code addresses or relates to the user's information need.

# Please provide your reasoning in a "Thought:" section, then conclude with "Result: X" where X is an integer between 0 and 10.

# # Scoring Guidelines:
# - 0-1: Completely irrelevant - code has no relation to the query topic
# - 2-3: Structurally relevant - code is related to the query topic but doesn't provide specific information
# - 4-5: Minimally relevant - code mentions related concepts but doesn't directly address the query
# - 6-7: Moderately relevant - code partially addresses the query or provides related functionality
# - 8-9: Highly relevant - code directly addresses most aspects of the query
# - 10: Perfectly relevant - code directly and comprehensively answers the query

# # Examples

# Query: How does user authentication work in this system?
# Code:
# ```
# # class method authenticate_user in auth/authentication.py
# def authenticate_user(self, username: str, password: str) -> bool:
#     \"\"\"
#     Authenticate a user by checking their credentials against the database.

#     Parameters
#     ----------
#     username : str
#         The username to authenticate
#     password : str
#         The password to check

#     Returns
#     -------
#     bool
#         True if authentication succeeds, False otherwise
#     \"\"\"
#     # Hash the provided password
#     hashed_password = self._hash_password(password)

#     # Query database for user
#     user = self.db.get_user(username)
#     if not user:
#         return False

#     # Compare hashed passwords
#     return user.password_hash == hashed_password
# ```
# Thought: The user is asking about how user authentication works in the system. The provided code is an authenticate_user method that shows the complete authentication flow: hashing passwords, querying the database for users, and comparing password hashes. This directly addresses the user's query by demonstrating the authentication mechanism step by step. The code is highly relevant as it shows exactly how authentication is implemented.
# Result: 10

# Query: What database models are available?
# Code:
# ```
# # top-level function calculate_tax in utils/financial.py
# def calculate_tax(income: float, tax_rate: float) -> float:
#     \"\"\"
#     Calculate tax amount based on income and tax rate.

#     Parameters
#     ----------
#     income : float
#         The income amount
#     tax_rate : float
#         The tax rate as a decimal (e.g., 0.15 for 15%)

#     Returns
#     -------
#     float
#         The calculated tax amount
#     \"\"\"
#     return income * tax_rate
# ```
# Thought: The user is asking about database models available in the system, but the provided code is a tax calculation function in a financial utilities module. This code has no relation to database models, schemas, or data structures. It's completely unrelated to the user's query about database models.
# Result: 0

# Query: How does data validation work?
# Code:
# ```
# # class method get_user_preferences in user/profile.py
# def get_user_preferences(self, user_id: int) -> dict:
#     \"\"\"
#     Retrieve user preferences from database.

#     Parameters
#     ----------
#     user_id : int
#         The ID of the user

#     Returns
#     -------
#     dict
#         User preferences as key-value pairs
#     \"\"\"
#     if not isinstance(user_id, int) or user_id <= 0:
#         raise ValueError("User ID must be a positive integer")

#     user = self.db.get_user(user_id)
#     if not user:
#         raise ValueError(f"User with ID {{user_id}} not found")

#     return user.preferences or {{}}
# ```
# Thought: The user is asking about how data validation works in the system. The provided code shows a method that retrieves user preferences, and it does include some basic validation (checking if user_id is a positive integer and if the user exists). However, this is minimal input validation rather than a comprehensive data validation system. The code only partially addresses the query by showing some validation examples, but doesn't demonstrate the broader validation framework or patterns used in the application.
# Result: 6

# # Now evaluate this query and code:
# Query: {query}
# Code:
# ```
# # {node_type}
# {code}
# ```
#         """
relevance_score_prompt = """
You are a code relevance evaluator who helps users find relevant code snippets for their queries about a software repository.
Your main responsibility is to examine user queries and determine how relevant a given piece of code is to answering that query.
Given a user query and a piece of code from the repository, please read and understand the query carefully, and determine how well the given code addresses or relates to the user's information need.

Since multiple code snippets will be retrieved and combined to answer the query, you should reward code that provides partial answers or demonstrates related functionality, as these contribute valuable context when combined with other relevant snippets.

Please provide your reasoning in a "Thought:" section, then conclude with "Result: X" where X is an integer between 0 and 10.

# Scoring Guidelines:
- 0-1: Completely irrelevant - code has no relation to the query topic
- 2-3: Tangentially relevant - code is loosely related to the query topic but provides minimal useful information
- 4-5: Partially relevant - code provides some useful information or demonstrates related concepts that contribute to understanding the query
- 6-7: Moderately relevant - code addresses significant aspects of the query or provides important related functionality  
- 8-9: Highly relevant - code directly addresses most aspects of the query or demonstrates key functionality
- 10: Perfectly relevant - code directly and comprehensively answers the query

# Additional Considerations:
- Functional code demonstrating behavior is often more valuable than documentation
- Structural details and implementation patterns reveal system capabilities
- Partial answers contribute value when multiple snippets are combined
- Consider implicit information from code structure and naming

# Examples

Query: How does user authentication work in this system?
Code:
```
# class method authenticate_user in auth/authentication.py
def authenticate_user(self, username: str, password: str) -> bool:
    \"\"\"
    Authenticate a user by checking their credentials against the database.
    
    Parameters
    ----------
    username : str
        The username to authenticate
    password : str  
        The password to check
        
    Returns
    -------
    bool
        True if authentication succeeds, False otherwise
    \"\"\"
    # Hash the provided password
    hashed_password = self._hash_password(password)
    
    # Query database for user
    user = self.db.get_user(username)
    if not user:
        return False
        
    # Compare hashed passwords
    return user.password_hash == hashed_password
```
Thought: This code directly shows the complete authentication process including password hashing, database querying, and credential verification. It fully addresses the user's query about authentication implementation.
Result: 10

Query: What database models are available?
Code:
```
# top-level function calculate_tax in utils/financial.py  
def calculate_tax(income: float, tax_rate: float) -> float:
    \"\"\"
    Calculate tax amount based on income and tax rate.
    
    Parameters
    ----------
    income : float
        The income amount
    tax_rate : float
        The tax rate as a decimal (e.g., 0.15 for 15%)
        
    Returns
    -------
    float
        The calculated tax amount
    \"\"\"
    return income * tax_rate
```
Thought: This is a tax calculation function unrelated to database models. No connection to the query topic.
Result: 0

Query: How does data validation work?
Code:
```
# class method get_user_preferences in user/profile.py
def get_user_preferences(self, user_id: int) -> dict:
    \"\"\"
    Retrieve user preferences from database.
    
    Parameters
    ----------
    user_id : int
        The ID of the user
        
    Returns
    -------
    dict
        User preferences as key-value pairs
    \"\"\"
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("User ID must be a positive integer")
        
    user = self.db.get_user(user_id)
    if not user:
        raise ValueError(f"User with ID {{user_id}} not found")
        
    return user.preferences or {{}}
```
Thought: This shows practical validation examples including type checking, range validation, and existence validation. Demonstrates validation patterns that contribute to understanding the system's validation approach.
Result: 6

# Now evaluate this query and code:
Query: {query}
Code:
```
# {node_type} 
{code}
```
"""

relevance_score_prompt_v2 = """
You are a code exploration evaluator. Given a user query and a node description (which could be a detailed function/class with purpose + members, or a high-level module/file description), determine how likely this node is to contain relevant information for the query. Score nodes that show potential for containing useful functionality, as the goal is identifying promising areas for deeper investigation.
Please provide your reasoning in a **BRIEF** "Thought:" section, then conclude with "Result: X" where X is an integer between 0 and 10.

# Scoring Guidelines:
* 0-1: Completely irrelevant - node and its members have no relation to the query topic and exploring it would be unproductive
* 2-3: Tangentially relevant - node and its members are loosely related but unlikely to contain substantial information for the query
* 4-5: Potentially relevant - node and its members shows some connection to the query topic and may contain useful information worth exploring
* 6-7: Likely relevant - node and its members appears to address significant aspects of the query and exploration is recommended
* 8-9: Highly relevant - node and its members strongly matches the query topic and should definitely be explored
* 10: Perfectly relevant - node and its members directly addresses the query topic and is essential to explore

# Additional Considerations:
* For detailed nodes: Purpose alignment and related members suggest high relevance
* For module descriptions: Look for functionality keywords that match query intent
* Consider both direct matches and contextually related functionality
* Abstract descriptions may hide valuable details worth exploring

# Examples

Query: How does the caching system work?
Description:
**Purpose:** Manages application configuration and environment settings.
**Members:** load_config() - Loads configuration from files, validate_settings() - Validates configuration parameters, cache_timeout - Timeout value for cache operations, redis_host - Redis server hostname, clear_cache() - Function to invalidate cached data
Thought: While the purpose focuses on configuration, the members reveal cache-related functionality including cache_timeout and clear_cache(), making this worth exploring for caching insights.
Result: 6

Query: How are API requests handled?
Description:
**Purpose:** Handles HTTP request processing and response management.
**Members:** request_handler() - Main request processing function, parse_headers() - Extracts HTTP headers, validate_json() - Validates JSON payloads, rate_limiter - Controls request frequency, error_formatter() - Formats error responses
Thought: Both purpose and members directly address API request handling with comprehensive request processing functionality.
Result: 9

Query: What payment processing features exist?
Description:
**Purpose:** Provides logging and monitoring capabilities for application events.
**Members:** log_error() - Records error messages, track_performance() - Monitors system performance, audit_trail - Maintains activity records, alert_admin() - Sends notifications to administrators
Thought: Purpose and members focus entirely on logging/monitoring with no connection to payment processing functionality.
Result: 1

Query: How does data validation work?
Description:
**Purpose:** Database connection and query execution utilities.
**Members:** connect_db() - Establishes database connection, execute_query() - Runs SQL queries, validate_schema() - Checks data against database schema, sanitize_input() - Cleans user input data, transaction_manager - Handles database transactions
Thought: Despite database focus, members include validate_schema() and sanitize_input() which are core validation functions relevant to the query.
Result: 7

# Now evaluate this query and node:
Query: {query}
Description:
{description}
"""


relevance_score_prompt_v3 = """
You are a code exploration evaluator. Given a user query and a node description (which could be a detailed function/class with purpose + members, or a high-level module/file description), determine how likely this node is to contain relevant information for the query. Score nodes that show potential for containing useful functionality, as the goal is identifying promising areas for deeper investigation.

Please provide your reasoning in a **BRIEF** "Thought:" section, then conclude with "Result: X" where X is an integer between 0 and 10. 

The output should **ONLY** be in the following format:
```json
{{
    "thought": "your BRIEF reasoning",
    "result": "integer between 0 and 10"
}}
```

# Scoring Guidelines:
* 0-1: Completely irrelevant - node and its members have no relation to the query topic and exploring it would be unproductive
* 2-3: Tangentially relevant - node and its members are loosely related but unlikely to contain substantial information for the query
* 4-5: Potentially relevant - node and its members shows some connection to the query topic and may contain useful information worth exploring
* 6-7: Likely relevant - node's members contain functionality that addresses significant aspects of the query and exploration is recommended
* 8: Very relevant - the node's members strongly match the query topic and should be explored
* 9: Highly relevant - the node itself strongly matches the query topic and has information to answer the query
* 10: Perfectly relevant - both the node itself and its members directly address the query topic and are essential to explore

**Priority System:**
- **Highest Priority (9-10):** Node itself is very useful AND its children/members are useful
- **High Priority (6-9):** Node itself (higher priority) OR its children/members are useful 
- **Standard Priority (0-5):** General relevance assessment based on overall connection to query

# Additional Considerations:
* For detailed nodes: Purpose alignment and related members suggest high relevance
* For module descriptions: Look for functionality keywords that match query intent
* Consider both direct matches and contextually related functionality
* Abstract descriptions may hide valuable details worth exploring
* **Prioritize nodes where members contain query-relevant functionality - useful members can elevate scores to 6-9 range**

# Examples
Query: How does the caching system work?
Description:
**Purpose:** Manages application configuration and environment settings.
**Members:** load_config() - Loads configuration from files, validate_settings() - Validates configuration parameters, cache_timeout - Timeout value for cache operations, redis_host - Redis server hostname, clear_cache() - Function to invalidate cached data
```json
{{
    "thought": "Node purpose is configuration-focused, but members contain valuable cache-related functionality (cache_timeout, clear_cache()) that directly addresses the caching query.",
    "result": "7"
}}
```

Query: How are API requests handled?
Description:
**Purpose:** Handles HTTP request processing and response management.
**Members:** request_handler() - Main request processing function, parse_headers() - Extracts HTTP headers, validate_json() - Validates JSON payloads, rate_limiter - Controls request frequency, error_formatter() - Formats error responses
```json
{{
    "thought": "Both the node's purpose (HTTP request processing) and its members (request_handler, parse_headers, etc.) directly and comprehensively address API request handling.",
    "result": "10"
}}
```

Query: What payment processing features exist?
Description:
**Purpose:** Provides logging and monitoring capabilities for application events.
**Members:** log_error() - Records error messages, track_performance() - Monitors system performance, audit_trail - Maintains activity records, alert_admin() - Sends notifications to administrators
```json
{{
    "thought": "Neither the node's purpose nor its members relate to payment processing functionality.",
    "result": "0"
}}
```

Query: How does user authentication work?
Description:
**Purpose:** Manages user session and activity tracking.
**Members:** track_login_time() - Records when users log in, session_duration() - Calculates session length, user_activity_log() - Logs user actions, last_active_timestamp - Stores last activity time, cleanup_expired_sessions() - Removes old session data
```json
{{
    "thought": "Node purpose involves user sessions which is authentication-adjacent, and members track login/session data that provides some insight into authentication flow, though not core auth mechanisms.",
    "result": "3"
}}
```

# Now evaluate this query and node:
Query: {query}
Description:
{description}
"""

repobench_query_generation_prompt = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Your task is to generate accurate Cypher queries for code dependency analysis based on a provided graph schema and user queries containing code snippets.

## Graph Schema
### Nodes Description:
{nodes_description}

### Edges Description:
{edges_description}

## Instructions:

1. **ONLY generate and return a Cypher query if it can be answered using the information provided in the schema above**

2. **CRITICAL REQUIREMENT: The user query will contain a code snippet along with file information. You must:**
   - First identify the node containing the given code using file_name and code context
   - Then fetch all dependencies (connected nodes) that are required for next line prediction
   - **MANDATORY: Return ONLY the 'code' and 'signature' attributes of the connected nodes**

3. **Do NOT make assumptions about node properties, relationships, or data that are not explicitly defined in the schema**

4. **Use proper Cypher syntax and follow Neo4j best practices**

5. **Focus on retrieving dependencies that would be essential for code completion/prediction**

6. For the MODULE node, use the 'name' property in case of dotted name, and use the 'local_name' property in case of undotted name.

## Example Queries:

### Example 1
**User Query:** 
```
Given file_name: utils/database.py 
Fetch dependencies for code: "class DatabaseConnection:"
```

**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'utils.database'}}-[:CONTAINS]->(c:Class {{name: 'DatabaseConnection'}})
OPTIONAL MATCH (c)-[:USES|:INHERITS_FROM|:HAS_METHOD]->(dep)
RETURN DISTINCT
  dep.signature,
  dep.code
```

### Example 2
**User Query:**
```
Given file_name: tests/main_test.py
Fetch dependencies for code:
def _args(**kwargs):
    kwargs.setdefault('command', 'help')
    kwargs.setdefault('config', C.CONFIG_FILE)
    return argparse.Namespace(**kwargs)

def test_adjust_args_and_chdir_not_in_git_dir(in_tmpdir):
```

**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'tests.main_test'}}-[:CONTAINS]->(f:Function)
WHERE f.name IN [
  'test_adjust_args_and_chdir_not_in_git_dir',
  '_args'
]
OPTIONAL MATCH (f)-[:USES|:INHERITS_FROM|:HAS_METHOD]->(dep)
RETURN DISTINCT
  dep.signature,
  dep.code
```

### Example 3
**User Query:**
```
Given file_name: handlers/auth.py
Fetch dependencies for code: "async def authenticate(request):"
```

**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'handlers.auth'}}-[:CONTAINS]->(f:Function {{name: 'authenticate'}})
OPTIONAL MATCH (f)-[:USES]->(dep)
WHERE dep:Class OR dep:Function OR dep:Method OR dep:GlobalVariable
RETURN DISTINCT
  dep.signature,
  dep.code
```

## Your Task:
Given the user query below containing a code snippet and file information, generate and return **ONLY THE CYPHER QUERY**, no other text or explanation. The query must identify the code node and fetch all its connected dependencies with their code and signature attributes.

**User Query:**
"""

repobench_query_generation_prompt_minimal = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Generate **CONCISE** Cypher queries for code dependency analysis based on the provided graph schema.
You are looking for **DEPENDENCIES** (what the incomplete code needs to use), NOT siblings

## Instructions:

1. **FOCUS ON THE BOTTOM INCOMPLETE CODE ONLY** - Don't query other methods/classes in the file
2. **GENERATE MINIMAL QUERIES** - Use the fewest UNION clauses possible
3. **FOR INCOMPLETE functions/classes/method/collections**: Identify ONLY the specific incomplete node, fetch ONLY its direct dependencies. DO NOT fetch file-level dependencies.
4. **ONLY FOR THE CASE OUTSIDE THOSE MENTIONED IN POINT 3**: Fetch the dependencies of the entire file.
5. **MANDATORY: Return ONLY 'name', 'code' and 'signature' attributes of connected nodes**
6. **ABSOLUTE RULE: Never fetch file dependencies for structured code elements (classes/functions/methods/collections). Only fetch the specific node's dependencies.**
7. **Use proper Cypher syntax. Ensure each UNION branch in Cypher has a complete MATCH...RETURN with identical column names and orders**
8. **Pay attention to the file path, so you do not miss any folder names. THIS IS VERY IMPORTANT**
9. **For MODULE nodes**: Use 'name' for dotted names, 'local_name' for undotted names.

## Graph Schema
### Nodes Description:
{nodes_description}

### Edges Description:
{edges_description}


## Example Queries:

### Example 1 - Incomplete Class with Method
**User Query:** 
```
Given file_name: src.alert.interference.reporting.admin.admin.py
Fetch dependencies for code:
class ColumnTemplateAdmin(DataspacedAdmin): 
  def get_form_class(self, request, obj=None):
        return ColumnTemplateForm(request)
  def get_client_data(self, request):
```
**Cypher Query:**
```cypher

MATCH (m:Module {{name: 'src.alert.interference.reporting.admin.admin'}})-[:CONTAINS]->(c:Class {{name: 'ColumnTemplateAdmin'}})-[:HAS_METHOD]->(method)
WHERE method.name = 'get_client_data'
OPTIONAL MATCH (method)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 2 - Incomplete Function
**User Query:**
```
Given file_name: projects.Detic_new.detic.detic.py
Fetch dependencies for code:
    if not isinstance(data, list):
        raise ValueError("Input must be a list")
    return True

def transform_item(item):
    return {{
        'id': item.get('id'),
        'value': item.get('value', '').strip(),
        'timestamp': datetime.now()
    }}

@my_decorator
def process_data(items):
    for item in items:
```
**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'projects.Detic_new.detic.detic'}})-[:CONTAINS]->(f:Function {{name: 'process_data'}})
OPTIONAL MATCH (f)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 3 - Unstructured Global-level Code Context **YOU WILL OBTAIN THE DEPENDENCIES OF THE ENTIRE FILE**
**User Query:** 
```
Given file_name: source.tools.comparison.py
Fetch dependencies for code:
except Exception as eee:
    raise OSError(error)
v4_list = dir(IPv4Obj("127.0.0.1"))
```
**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'source.tools.comparison'}})
MATCH (m)-[:CONTAINS]->(node)
OPTIONAL MATCH (m)-[:CONTAINS]->(c:Class)-[:HAS_METHOD]->(method:Method)
WITH collect(node) + collect(method) as all_nodes
UNWIND all_nodes as n
MATCH (n)-[:USES]->(dep)
WHERE dep.module_name <> 'source.tools.comparison'
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```
### Example 4 - Incomplete Global COLLECTIONS like lists, dictionaries, etc.
**User Query:**
```
Given file_name: main.utils.config.settings.py
Fetch dependencies for code:
ALLOWED_HOSTS = [
'localhost',
```
**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'main.utils.config.settings'}})-[:CONTAINS]->(g:GlobalVariable {{name: 'ALLOWED_HOSTS'}})
OPTIONAL MATCH (g)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

## Your Task:
Given the user query below containing a code snippet and file information, generate and return **ONLY THE CYPHER QUERY**, no other text or explanation. 

**FOCUS ON THE BOTTOM INCOMPLETE CODE ONLY**

**User Query:**

"""


repobench_query_generation_prompt_v2 = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Your task is to generate accurate Cypher queries for code dependency analysis based on a provided graph schema and user queries containing code snippets.

## Graph Schema
### Nodes Description:
{nodes_description}

### Edges Description:
{edges_description}

## Instructions:

1. **ONLY generate and return a Cypher query if it can be answered using the information provided in the schema above**


2. **CRITICAL REQUIREMENT: The user query will contain a code snippet along with file information. You must:**
   - **For incomplete functions/classes**: First identify the node containing the givenincomplete code, then fetch its dependencies. IMPORTANT: Only fetch the incomplete code node and its dependencies NOT nodes of all the complete code before it.
   - **For global-level code**: Analyze the code context to predict what identifiers/symbols are likely needed for the next line, then fetch those specific nodes
   - **MANDATORY: Return ONLY the 'name', 'code' and 'signature' attributes of the connected nodes**

3. **Do NOT make assumptions about node properties, relationships, or data that are not explicitly defined in the schema**

4. **Use proper Cypher syntax and follow Neo4j best practices, like if using UNION ensure that the return columns have the same name**

5. **Focus on retrieving dependencies that would be essential for code completion/prediction**

6. For the MODULE node, use the 'name' property in case of dotted name, and use the 'local_name' property in case of undotted name.


## Example Queries:

### Example 1
**User Query:** 
```
Given file_name: utils/database.py 
Fetch dependencies for code: "class DatabaseConnection(BaseConnection):"
```

**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'utils.database'}}-[:CONTAINS]->(c:Class {{name: 'DatabaseConnection'}})
OPTIONAL MATCH (c)-[:USES|INHERITS]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
UNION
MATCH (m:Module {{name: 'utils.database'}}-[:CONTAINS]->(c:Class {{name: 'DatabaseConnection'}})
MATCH (c)-[:HAS_METHOD]->(method)
OPTIONAL MATCH (method)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
UNION
MATCH (m:Module {{name: 'utils.database'}}-[:CONTAINS]->(c:Class {{name: 'BaseConnection'}})
OPTIONAL MATCH (c)-[:USES|INHERITS]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
UNION
MATCH (m:Module {{name: 'utils.database'}}-[:CONTAINS]->(c:Class {{name: 'BaseConnection'}})
MATCH (c)-[:HAS_METHOD]->(method)
OPTIONAL MATCH (method)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 2
**User Query:**
```
Given file_name: tests/main_test.py
Fetch dependencies for code:
def _args(**kwargs):
    kwargs.setdefault('command', 'help')
    kwargs.setdefault('config', C.CONFIG_FILE)
    return argparse.Namespace(**kwargs)

def test_adjust_args_and_chdir_not_in_git_dir(in_tmpdir):
```

**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'tests.main_test'}})-[:CONTAINS]->(f:Function)
WHERE f.name = 'test_adjust_args_and_chdir_not_in_git_dir'
OPTIONAL MATCH (f)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 3
**User Query:**
```
Given file_name: tools/comparison.py
Fetch dependencies for code:
# Compare methods on IPv4Obj() and IPv6Obj(). Flag missing methods
sys.path.insert(0, "../")
environ = os.environ['VIRTUAL_ENV']
print("ENV", environ)
try:
    print("PYTHONPATH", str(os.environ['PYTHONPATH']))
except Exception as eee:
    error = f"{{eee}}: Could not find PYTHONPATH."
    logger.error(error)
    raise OSError(error)
v4_list = dir(IPv4Obj("127.0.0.1"))
```

**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'tools.comparison'}})-[:CONTAINS]->(contained)
OPTIONAL MATCH (contained)-[:USES]->(neighbor)
WHERE neighbor.name IN ['IPv6Obj', 'dir'] 
   OR neighbor.signature CONTAINS 'IPv6Obj'
   OR neighbor.code CONTAINS 'IPv6Obj'
RETURN neighbor.name AS name, neighbor.signature AS signature, neighbor.code AS code
UNION
MATCH (m:Module {{name: 'tools.comparison'}})-[:CONTAINS]->(func)
WHERE func.name CONTAINS 'IPv6' OR func.signature CONTAINS 'IPv6'
RETURN func.name AS name, func.signature AS signature, func.code AS code

```

### Example 4
**User Query:**
```
Given file_name: network/ios_factory.py  
Fetch dependencies for code:
ALL_IOS_FACTORY_CLASSES = [   
    IOSIntfLine,      # -> USES relationship to IOSIntfLine class
    IOSRouteLine,
]
```
**Cypher Query:**
```cypher
MATCH (m:Module {{name: 'network.ios_factory'}})-[:CONTAINS]->(g:GlobalVariable {{name: 'ALL_IOS_FACTORY_CLASSES'}})
OPTIONAL MATCH (g)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code

```
## Your Task:
Given the user query below containing a code snippet and file information, generate and return **ONLY THE CYPHER QUERY**, no other text or explanation. The query must identify the code node and fetch all its connected dependencies with their name, code and signature attributes.

**User Query:**
"""

repobench_query_generation_prompt_concise_backup = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Your task is to generate concise Cypher queries to find dependencies for incomplete code snippets based on the provided graph schema.

## Instructions:

1. **IDENTIFY THE INCOMPLETE ELEMENT NEAR THE BOTTOM OF THE CODE SNIPPET**
   - **For collections/lists/dicts**: Missing closing bracket `]`, `}}`, or `)` 
   - **For functions/classes/methods**: Missing body, incomplete signature, or abrupt termination

2. **If NOTHING is INCOMPLETE, then use the global fallback approach**

3. **FOR INCOMPLETE structured code elements like functions/classes/method/collections**: Carefully identify the name of the incomplete element, then get the specific incomplete node, and finallyfetch ONLY its direct dependencies. DO NOT fetch file-level dependencies.

4. **When code is complete or no specific incomplete structured element can be identified with high confidence**: Use the global fallback approach

5. **GENERATE MINIMAL QUERIES** - Use the fewest UNION clauses possible

6. **MANDATORY: Return ONLY 'name', 'code' and 'signature' attributes of connected nodes**

7. **Use proper Cypher syntax. For UNION, ensure return columns have same names**

8. **Pay attention to the file path, so you do not miss any folder names. THIS IS VERY IMPORTANT**

9. **For MODULE nodes**: Use 'name' for dotted names, 'local_name' for undotted names.


## Graph Schema
nodes_description = 
**Module**: name (dotted), local_name, code, signature
**Class**: name, code, signature, module_name  
**Function**: name, code, signature, module_name
**Method**: name, code, signature, module_name, class
**Field**: name, class
**GlobalVariable**: name, code, module_name

edges_description =
**CONTAINS**: Module → Class/Function/GlobalVariable
**HAS_METHOD**: Class → Method
**HAS_FIELD**: Class → Field  
**INHERITS**: Class → Class
**USES**: Class/Function/Method/GlobalVariable → GlobalVariable/Field/Class/Function/Import

## Query Templates ** Read them carefully**:

**Incomplete Class Method:**
```cypher
MATCH (m:Module {{name: 'MODULE_NAME'}})-[:CONTAINS]->(c:Class {{name: 'CLASS_NAME'}})-[:HAS_METHOD]->(method {{name: 'METHOD_NAME'}})
OPTIONAL MATCH (method)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

**Incomplete Function:**
```cypher
MATCH (m:Module {{name: 'MODULE_NAME'}})-[:CONTAINS]->(f:Function {{name: 'FUNCTION_NAME'}})
OPTIONAL MATCH (f)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

**Incomplete Global Collection:**
```cypher
MATCH (m:Module {{name: 'MODULE_NAME'}})-[:CONTAINS]->(g:GlobalVariable {{name: 'COLLECTION_NAME'}})
OPTIONAL MATCH (g)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

**Fallback – Global Code:** 
```cypher
MATCH (m:Module {{name: 'MODULE_NAME'}})
MATCH (m)-[:CONTAINS]->(node)
OPTIONAL MATCH (m)-[:CONTAINS]->(c:Class)-[:HAS_METHOD]->(method:Method)
WITH collect(node) + collect(method) AS all_nodes
UNWIND all_nodes AS n
MATCH (n)-[:USES]->(dep)
WHERE dep.module_name <> 'MODULE_NAME'
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

## Your Task:
Given the user query below containing a code snippet and file information, generate and return **ONLY THE CYPHER QUERY**, no other text or explanation. 

**Key Decision:** Is there an incomplete DEFINITION near the bottom? If not, use fallback approach.

## User Query

"""


repobench_query_generation_prompt_concise_backup_good = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Your task is to generate concise Cypher queries to find dependencies for code snippets based on the provided graph schema.

## Decision Process:
1. **ANALYZE CODE COMPLETENESS**: Check if there's an incomplete element near the bottom of the code snippet
2. **IF COMPLETE**: Use global fallback approach for file-level dependencies  
3. **TO FIND INCOMPLETE**: 
  - 3a **For collections/lists/dicts**: Missing closing bracket `]`, `}}`, or `)` 
  - 3b **For functions/classes/methods**: Missing body, incomplete signature, or abrupt termination
4. **IMPORTANT**: Only use visible information. DO NOT GUESS incomplete elements if their definitions aren't clearly shown.

## Instructions:
- **CRITICAL**: When you find an incomplete function/method/class/collection, you MUST identify its name and use the specific template for that element
- **GENERATE MINIMAL QUERIES**: Use fewest UNION clauses possible
- **MANDATORY**: Return ONLY 'name', 'code', 'signature' attributes
- **IMPORTANT**: Pay attention to complete file path including folder names
- **MODULE NODES**: Use 'name' for dotted names, 'local_name' for undotted names
- **INDENTATION**: Pay attention to the indentation of the code snippet to identify if the incomplete element is a method or a function.

## Graph Schema:
**Nodes**: Module (name, local_name, code, signature), Class (name, code, signature, module_name), Function (name, code, signature, module_name), Method (name, code, signature, module_name, class), Field (name, class), GlobalVariable (name, code, module_name)

**Edges**: CONTAINS (Module→Class/Function/GlobalVariable), HAS_METHOD (Class→Method), HAS_FIELD (Class→Field), INHERITS (Class→Class), USES (All→Dependencies)

## Example Queries:

### Example 1 - Incomplete Method
**User Query:**
```
Given file_name: src.alert.interference.reporting.admin.admin.py
Fetch dependencies for code:
    def get_form_class(self, request, obj=None):
        return ColumnTemplateForm(request)
    def get_client_data(self, request):
```

**Thought:** Incomplete element identified: method `get_client_data`.

**Query:**
```cypher
MATCH (m:Module {{name: 'src.alert.interference.reporting.admin.admin'}})-[:CONTAINS]->(c:Class {{name: 'ColumnTemplateAdmin'}})-[:HAS_METHOD]->(method {{name: 'get_client_data'}})
OPTIONAL MATCH (method)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 2 - Incomplete Function
**User Query:**
```
Given file_name: projects.Detic_new.detic.detic.py
Fetch dependencies for code:
def transform_item(item):
    return {{
        'id': item.get('id'),
        'value': item.get('value', '').strip(),
        'timestamp': datetime.now()
    }}

@my_decorator
def process_data(items):
    for item in items:
```

**Thought:** Incomplete element identified: function `process_data`.

**Query:**
```cypher
MATCH (m:Module {{name: 'projects.Detic_new.detic.detic'}})-[:CONTAINS]->(f:Function {{name: 'process_data'}})
OPTIONAL MATCH (f)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 3 - Complete/Unstructured Code
**User Query:**
```
Given file_name: source.tools.comparison.py
Fetch dependencies for code:
except Exception as eee:
    raise OSError(error)
v4_list = dir(IPv4Obj("127.0.0.1"))
```

**Thought:** No clear incomplete element found; avoiding guessing, using global fallback

**Query:**
```cypher
MATCH (m:Module {{name: 'source.tools.comparison'}})
MATCH (m)-[:CONTAINS]->(node)
OPTIONAL MATCH (m)-[:CONTAINS]->(c:Class)-[:HAS_METHOD]->(method:Method)
WITH collect(node) + collect(method) AS all_nodes
UNWIND all_nodes AS n
MATCH (n)-[:USES]->(dep)
WHERE dep.module_name <> 'source.tools.comparison'
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 4 - Incomplete Global Variable
**User Query:**
```
Given file_name: main.utils.config.settings.py
Fetch dependencies for code:
ALLOWED_HOSTS = [
    'localhost',
```

**Thought:** Incomplete element identified: global variable `ALLOWED_HOSTS`.

**Query:**
```cypher
MATCH (m:Module {{name: 'main.utils.config.settings'}})-[:CONTAINS]->(g:GlobalVariable {{name: 'ALLOWED_HOSTS'}})
OPTIONAL MATCH (g)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

## Your Task:
First provide a brief thought on your decision process, then generate **ONLY THE CYPHER QUERY**.

**Format:**
```
**Thought:** [Incomplete element identified: <element_name> OR No incomplete element identified]
**Query:**
[Cypher query only]
```

## User Query:s
"""

repobench_query_generation_prompt_concise = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Your task is to generate concise Cypher queries to find dependencies for code snippets based on the provided graph schema.

## Decision Process:
1. **ANALYZE CODE COMPLETENESS**: Check if there's an incomplete element near the bottom of the code snippet
2. **IF COMPLETE**: Use global fallback approach for file-level dependencies  
3. **TO FIND INCOMPLETE**: 
  - 3a **For collections/lists/dicts**: Missing closing bracket `]`, `}}`, or `)` 
  - 3b **For functions/classes/methods**: Missing body, incomplete signature, or abrupt termination
4. **CRITICAL**: Only use visible information. DO NOT GUESS incomplete elements if their definitions aren't clearly shown. **NEVER ASSUME** - if unsure, always use global fallback.
   - **ONLY** identify incomplete elements if you see actual `def`, `class`, or variable assignment with collections `[`, `{{`
   - Don't identify based on function calls/usage or comments

## Instructions:
- **CRITICAL**: When you find an incomplete function/method/class/collection, you MUST identify its name and use the specific template for that element - BUT ONLY if the definition is clearly visible
- **NO GUESSING**: If the element definition is not clearly shown, use global fallback instead
- **MUST SEE**: Actual `def function():`, `class Name:`, or `variable = [` syntax to identify incomplete elements
- **INDENTATION MATTERS**: Pay close attention to indentation to distinguish functions (no indent) vs methods (indented) - this is crucial for correct queries
- **GENERATE MINIMAL QUERIES**: Use fewest UNION clauses possible
- **MANDATORY**: Return ONLY 'name', 'code', 'signature' attributes
- **IMPORTANT**: Pay attention to complete file path including folder names
- **MODULE NODES**: Use 'name' for dotted names, 'local_name' for undotted names
- **CORRECTNESS**: Use proper Cypher syntax. Ensure each UNION branch in Cypher has a complete MATCH...RETURN with identical column names and orders


## Graph Schema:
**Nodes**: Module (name, local_name, code, signature), Class (name, code, signature, module_name), Function (name, code, signature, module_name), Method (name, code, signature, module_name, class), Field (name, class), GlobalVariable (name, code, module_name)

**Edges**: CONTAINS (Module→Class/Function/GlobalVariable), HAS_METHOD (Class→Method), HAS_FIELD (Class→Field), INHERITS (Class→Class), USES (All→Dependencies)

## Example Queries:

### Example 1 - Incomplete Method
**User Query:**
```
Given file_name: src.alert.interference.reporting.admin.admin
Fetch dependencies for code:
    def get_form_class(self, request, obj=None):
        return ColumnTemplateForm(request)
    def get_client_data(self, request):
```

**Thought:** Incomplete element identified: method `get_client_data` (based on indentation).

**Query:**
```cypher
MATCH (m:Module {{name: 'src.alert.interference.reporting.admin.admin'}})-[:CONTAINS]->(c:Class {{name: 'ColumnTemplateAdmin'}})-[:HAS_METHOD]->(method {{name: 'get_client_data'}})
OPTIONAL MATCH (method)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 2 - Incomplete Function
**User Query:**
```
Given file_name: projects.Detic_new.detic.detic
Fetch dependencies for code:
def transform_item(item):
    return {{
        'id': item.get('id'),
        'value': item.get('value', '').strip(),
        'timestamp': datetime.now()
    }}

@my_decorator
def process_data(items):
    for item in items:
```

**Thought:** Incomplete element identified: function `process_data`.

**Query:**
```cypher
MATCH (m:Module {{name: 'projects.Detic_new.detic.detic'}})-[:CONTAINS]->(f:Function {{name: 'process_data'}})
OPTIONAL MATCH (f)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 3 - Complete/Unstructured Code
**User Query:**
```
Given file_name: source.tools.comparison
Fetch dependencies for code:
except Exception as eee:
    raise OSError(error)
v4_list = dir(IPv4Obj("127.0.0.1"))
```

**Thought:** No clear incomplete element found; definition not visible, avoiding guessing, using global fallback

**Query:**
```cypher
MATCH (m:Module {{name: 'source.tools.comparison'}})
MATCH (m)-[:CONTAINS]->(node)
OPTIONAL MATCH (m)-[:CONTAINS]->(c:Class)-[:HAS_METHOD]->(method:Method)
WITH collect(node) + collect(method) AS all_nodes
UNWIND all_nodes AS n
MATCH (n)-[:USES]->(dep)
WHERE dep.module_name <> 'source.tools.comparison'
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

### Example 4 - Incomplete Global Variable
**User Query:**
```
Given file_name: main.utils.config.settings
Fetch dependencies for code:
ALLOWED_HOSTS = [
    'localhost',
```

**Thought:** Incomplete element identified: global variable `ALLOWED_HOSTS`.

**Query:**
```cypher
MATCH (m:Module {{name: 'main.utils.config.settings'}})-[:CONTAINS]->(g:GlobalVariable {{name: 'ALLOWED_HOSTS'}})
OPTIONAL MATCH (g)-[:USES]->(dep)
RETURN DISTINCT dep.name AS name, dep.signature AS signature, dep.code AS code
```

## Your Task:
First provide a brief thought on your decision process, then generate **ONLY THE CYPHER QUERY**.

**Format:**
```
**Thought:** [Incomplete element identified: <element_name> OR No incomplete element identified]
**Query:**
[Cypher query only]
```

## User Query:
"""


crosscode_query_generation_prompt_concise = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Your task is to generate concise Cypher queries to find dependencies for incomplete functions or methods at the bottom of code snippets based on the provided graph schema.

## Decision Process:
1. **ANALYZE CODE**: The dataset contains code snippets with incomplete elements at the bottom
2. **IDENTIFY INCOMPLETE ELEMENT**: Look for the bottommost incomplete function or method which has definition
3. **FOR FUNCTIONS**: Partial implementation, missing logic, or abrupt termination with no indentation
4. **FOR METHODS**: Partial implementation, missing logic, or abrupt termination with indentation (inside a class)

## Instructions:
- **FOCUS**: Always identify the incomplete function or method at the bottom of the code snippet
- **INDENTATION MATTERS**: Pay close attention to indentation to distinguish functions (no indent) vs methods (indented) - this is crucial for correct queries
- **GENERATE MINIMAL QUERIES**: Use fewest UNION clauses possible
- **MANDATORY**: Return the enitre nodes and their labels in the query
- **IMPORTANT**: Use repo name and file name in the initial match pattern
- **MODULE NODES**: Use 'name' for dotted names, 'local_name' for undotted names
- **CORRECTNESS**: Use proper Cypher syntax. Ensure each UNION branch in Cypher has a complete MATCH...RETURN with identical column names and orders

## Graph Schema:
**Nodes**: Repo (name), Module (name, local_name, code, signature), Class (name, code, signature, module_name), Function (name, code, signature, module_name), Method (name, code, signature, module_name, class), Field (name, class)

**Edges**: CONTAINS (Repo→Module, Module→Class/Function), HAS_METHOD (Class→Method), HAS_FIELD (Class→Field), INHERITS (Class→Class), USES (All→Dependencies)

## Example Queries:

### Example 1 - Incomplete Method
**User Query:**
```
Given repo_name: /Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system
Given file_name: admin
Fetch the most important connected nodes from the graph to predict the next line of the below code:
    def get_form_class(self, request, obj=None):
        return ColumnTemplateForm(request)
    def get_client_data(self, request):
        data = {{'timestamp': datetime.now()}}
        client_info = request.META.get('HTTP_USER_AGENT')
```

**Thought:** Incomplete element identified: method `get_client_data` (based on indentation).

**Query:**
```cypher
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module {{name: 'admin'}})-[:CONTAINS]->(c:Class)-[:HAS_METHOD]->(method {{name: 'get_client_data'}})
OPTIONAL MATCH (method)-[:USES]->(dep)
WHERE dep IS NOT NULL
RETURN DISTINCT dep, labels(dep)
UNION
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module {{name: 'admin'}})-[:CONTAINS]->(c:Class)-[:HAS_METHOD]->(method {{name: 'get_client_data'}})
OPTIONAL MATCH (c)-[:INHERITS]->(parentClass:Class)-[:HAS_METHOD]->(dep)
RETURN DISTINCT dep, labels(dep)
```

### Example 2 - Incomplete Function
**User Query:**
```
Given repo_name: /Users/pratik.shah1/work/CrossCodeEval_repos/Detic_project
Given file_name: projects.Detic_new.detic.detic
Fetch the most important connected nodes from the graph to predict the next line of the below code:
def transform_item(item):
    return {{
        'id': item.get('id'),
        'value': item.get('value', '').strip(),
        'timestamp': datetime.now()
    }}

@my_decorator
def process_data(items):
    results = []
    for item in items:
        transformed = transform_item(item)
```

**Thought:** Incomplete element identified: function `process_data`.

**Query:**
```cypher
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/Detic_project'}})-[:CONTAINS]->(m:Module {{name: 'projects.Detic_new.detic.detic'}})-[:CONTAINS]->(f:Function {{name: 'process_data'}})
OPTIONAL MATCH (f)-[:USES]->(dep)
RETURN DISTINCT dep, labels(dep)
```

### Example 3 - Incomplete Method in Class
**User Query:**
```
Given repo_name: /Users/pratik.shah1/work/CrossCodeEval_repos/Azure_web_framework
Given file_name: models.views
Fetch the most important connected nodes from the graph to predict the next line of the below code:
class UserView:
    def __init__(self, request):
        self.request = request
        
    def get_context(self):
        return {{'user': self.request.user}}
        
    def render_template(self, template_name):
        context = self.get_context()
        template = loader.get_template(template_name)
```

**Thought:** Incomplete element identified: method `render_template`.

**Query:**
```cypher
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/Azure_web_framework'}})-[:CONTAINS]->(m:Module {{name: 'models.views'})-[:CONTAINS]->(c:Class)-[:HAS_METHOD]->(method {{name: 'render_template'}})
OPTIONAL MATCH (method)-[:USES]->(dep)
RETURN DISTINCT dep, labels(dep)
UNION
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/Azure_web_framework'}})-[:CONTAINS]->(m:Module {{name: 'models.views'}})-[:CONTAINS]->(c:Class)-[:HAS_METHOD]->(method {{name: 'render_template'}})
OPTIONAL MATCH (c)-[:INHERITS]->(parentClass:Class)-[:HAS_METHOD]->(dep)
RETURN DISTINCT dep, labels(dep)
```

## Your Task:
First provide a brief thought on your decision process, then generate **ONLY THE CYPHER QUERY**.

**Format:**
```
**Thought:** Incomplete element identified: <element_name> (function/method)
**Query:**
[Cypher query only]
```

## User Query:
"""
# crosscode_query_generation_prompt_v2 = """
# # Neo4j Cypher Query Expert for Code Dependency Analysis

# You are a Neo4j Cypher query expert. Your task is to generate concise Cypher queries to find ALL cross file dependencies that will help to complete the provided incomplete code based on the provided graph schema.

# ## Graph Schema:
# **Nodes**: Repo (name), Module, Class (name, code, signature, module_name), Function (name, code, signature, module_name), Method (name, code, signature, module_name, class)

# **Edges**: CONTAINS (Repo→Module, Module→Class/Function), HAS_METHOD (Class→Method), INHERITS (Class→Class), USES (All→Dependencies)

# #Instructions:
# - **CORRECTNESS**: Use proper Cypher syntax. Ensure each UNION branch in Cypher has a complete MATCH...RETURN with SAME COLUMN NAMES.!
# - **GENERATE MINIMAL QUERIES**:  ONLY RETIREVE THOSE NODES THAT YOU WILL REQUIRE TO COMPLETE THE INCOMPLETE CODE. Use fewest UNION clauses possible.
# - **MANDATORY**: Return the enitre nodes as ***dep*** and their labels as ***label*** in the query. NOTE THE NAMES SHOULD BE 'dep' and 'label' ONLY.
# - **IMPORTANT**: PAY EXTRA ATTENTION TO THE LAST INCOMPLETE LINE, THE FUNCTION/METHOD/CLASS BEING USED IN THE LAST INCOMPLETE LINE, AND TRACE THEM TO WHERE THEY ARE INSTATNIATED/IMPORTED, TO FETCH CORRECT DEPENDENCIES.
# - **IMPORTANT**: PAY EXTRA ATTENTION TO IMPORTS ALIASES, AND ONLY THE GLOBAL VARIABLES BEING USED IN THE LAST INCOMPLETE LINE.
# - **IMPORTANT**: In the generated cypher query ONLY USE NAMES YOU ARE CONFIDENT ABOUT OR ELSE DON'T USE THEM.  For imports, avoid module names as they may differ. It is fine if we get some false positives.
# - **IMPORTANT**: PAY ATTENTION TO THE PROVIDED GRAPH SCHEMA TO MAKE CORRECT QUERIES.


# ## Input Data Format:
# Given repo_name: Repository name which can use to identify the Repo Node in the graph.
# Given file_name: File name which can use to identify the Module Node in the graph.
# Fetch the most important connected nodes from the graph to predict the next line of the below code:
# Incomplete code snippet to complete.

# ## Your Task:
# First provide a brief thought on your decision process, then generate **ONLY THE CYPHER QUERY**.

# **Format:**
# ```
# **Thought:** Incomplete element identified: <element_name> (function/method)
# **Query:**
# [Cypher query only]

# ## Example
# Given repo_name: /Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system
# Given file_name: models.classes
# Fetch the most important connected nodes from the graph to predict the next line of the below code:
# import numpy as np
# from poptransformer import ops
# from poptransformer.layers.layer_norm import BaseLayerNorm
# from classes import BaseModule as base_module

# class BaseRMSLayerNorm(BaseLayerNorm):
#     def __init__(self, input_size, eps=1e-5, context=''):
#         self.base_object = base_module()

#     def collect_bind_layer_weights(self):
#         weight_key = '.'.join([self.context, 'weight'])
#         weight_np = self.get_param_from_state_dict(weight_key, [self.input_size])
#         self.weight_id = self.add_initialized_input_tensor(weight_np, weight_key)

#     def __call__(self, graph, x):
#         variance_epsilon = ops.constant(graph, np.array(self.eps).astype(np.float32), 'variance_epsilon')
#         variance = self.base_object.

# **Thought:**  Incomplete method __call__ in BaseRMSLayerNorm class, remaining methods are not important. The last incomplete line uses self.base_object, which calls base_module but that is an ALIAS of the imported BaseModule class suggesting need for BaseModule methods. Also need parent class BaseLayerNorm methods for inheritance context.

# **Query:**
# ```cypher
# MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseRMSLayerNorm'}})-[:HAS_METHOD]->(method {{name: '__call__'}})-[:USES]->(dep)
# RETURN DISTINCT dep, labels(dep) as label
# UNION
# MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseModule'}})-[:HAS_METHOD]->(method)
# RETURN DISTINCT method as dep, labels(method) as label
# UNION
# MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseLayerNorm'}})-[:HAS_METHOD]->(method)
# RETURN DISTINCT method as dep, labels(method) as label
# ```


# """
crosscode_query_generation_prompt_v2 = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Your task is to generate concise Cypher queries to find ALL cross file dependencies that will help to complete the provided incomplete code based on the provided graph schema.

## Graph Schema:
**Nodes**: Repo (name), Module, Class (name, code, signature, module_name), Function (name, code, signature, module_name), Method (name, code, signature, module_name, class)

**Edges**: CONTAINS (Repo→Module, Module→Class/Function), HAS_METHOD (Class→Method), INHERITS (Class→Class), USES (All→Dependencies)

#Instructions:
- **CORRECTNESS**: Use proper Cypher syntax. Ensure each UNION branch in Cypher has a complete MATCH...RETURN with SAME COLUMN NAMES! IMPORTANT TO HAVE SAME COLUMN NAMES FOR EACH UNION BRANCH.
- **GENERATE MINIMAL QUERIES**:  ONLY RETIREVE THOSE NODES THAT YOU WILL REQUIRE TO COMPLETE THE INCOMPLETE CODE. Use fewest UNION clauses possible.
- **MANDATORY**: Return the enitre nodes as ***dep*** and their labels as ***label*** in the query. NOTE THE NAMES SHOULD BE 'dep' and 'label' ONLY.
- **IMPORTANT**: PAY EXTRA ATTENTION TO THE LAST INCOMPLETE LINE, THE FUNCTION/METHOD/CLASS BEING USED IN THE LAST INCOMPLETE LINE, AND TRACE THEM TO WHERE THEY ARE INSTATNIATED/IMPORTED, TO FETCH CORRECT DEPENDENCIES.
- **IMPORTANT**: PAY EXTRA ATTENTION TO IMPORTS ALIASES, AND ONLY THE GLOBAL VARIABLES BEING USED IN THE LAST INCOMPLETE LINE.
- **IMPORTANT**: In the generated cypher query ONLY USE NAMES YOU ARE CONFIDENT ABOUT OR ELSE DON'T USE THEM.  For imports, avoid module names as they may differ. It is fine if we get some false positives.
- **IMPORTANT**: PAY ATTENTION TO THE PROVIDED GRAPH SCHEMA TO MAKE CORRECT QUERIES.


## Input Data Format:
Given repo_name: Repository name which can use to identify the Repo Node in the graph.
Given file_name: File name which can use to identify the Module Node in the graph.
Fetch the most important connected nodes from the graph to predict the next line of the below code:
Incomplete code snippet to complete.

## Your Task:
First provide a brief thought on your decision process, then generate **ONLY THE CYPHER QUERY**.

**Format:**
```
**Thought:** Incomplete element identified: <element_name> (function/method)
**Query:**
[Cypher query only]

## Example
Given repo_name: /Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system
Given file_name: models.classes
Fetch the most important connected nodes from the graph to predict the next line of the below code:
import numpy as np
from poptransformer import ops
from poptransformer.layers.layer_norm import BaseLayerNorm
from classes import BaseModule as base_module

class BaseRMSLayerNorm(BaseLayerNorm):
    def __init__(self, input_size, eps=1e-5, context=''):
        self.base_object = base_module()

    def collect_bind_layer_weights(self):
        weight_key = '.'.join([self.context, 'weight'])
        weight_np = self.get_param_from_state_dict(weight_key, [self.input_size])
        self.weight_id = self.add_initialized_input_tensor(weight_np, weight_key)

    def __call__(self, graph, x):
        variance_epsilon = ops.constant(graph, np.array(self.eps).astype(np.float32), 'variance_epsilon')
        variance = self.base_object.

**Thought:**  Incomplete method __call__ in BaseRMSLayerNorm class, remaining methods are not important. The last incomplete line uses self.base_object, which calls base_module but that is an ALIAS of the imported BaseModule class suggesting need for BaseModule. Also need parent class BaseLayerNorm for inheritance context.

**Query:**
```cypher
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseRMSLayerNorm'}})-[:HAS_METHOD]->(method {{name: '__call__'}})-[:USES]->(dep)
RETURN DISTINCT dep, labels(dep) as label
UNION
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseModule'}})
RETURN DISTINCT c as dep, labels(c) as label
UNION
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseLayerNorm'}})
RETURN DISTINCT c as dep, labels(c) as label
```


"""

# crosscode_query_generation_prompt_v2 = """
# # Neo4j Cypher Query Expert for Code Dependency Analysis

# You are a Neo4j Cypher query expert. Your task is to generate concise Cypher queries to find ALL cross file dependencies that will help to complete the provided incomplete code based on the provided graph schema.

# ## Graph Schema:
# **Nodes**: Repo (name), Module, Class (name, code, signature, module_name), Function (name, code, signature, module_name), Method (name, code, signature, module_name, class)

# **Edges**: CONTAINS (Repo→Module, Module→Class/Function), HAS_METHOD (Class→Method), INHERITS (Class→Class), USES (All→Dependencies)

# #Instructions:
# - **CORRECTNESS**: Use proper Cypher syntax. Ensure each UNION branch in Cypher has a complete MATCH...RETURN with SAME COLUMN NAMES.!
# - **GENERATE MINIMAL QUERIES**:  ONLY RETIREVE THOSE NODES THAT YOU WILL REQUIRE TO COMPLETE THE INCOMPLETE CODE. Use fewest UNION clauses possible.
# - **MANDATORY**: Return the enitre nodes as ***dep*** and their labels as ***label*** in the query. NOTE THE NAMES SHOULD BE 'dep' and 'label' ONLY.
# - **IMPORTANT**: PAY EXTRA ATTENTION TO THE LAST INCOMPLETE LINE, THE FUNCTION/METHOD/CLASS BEING USED IN THE LAST INCOMPLETE LINE, AND TRACE THEM TO WHERE THEY ARE INSTATNIATED/IMPORTED, TO FETCH CORRECT DEPENDENCIES.
# - **IMPORTANT**: PAY EXTRA ATTENTION TO IMPORTS ALIASES, AND ONLY THE GLOBAL VARIABLES BEING USED IN THE LAST INCOMPLETE LINE.
# - **IMPORTANT**: In the generated cypher query ONLY USE NAMES YOU ARE CONFIDENT ABOUT OR ELSE DON'T USE THEM.  For imports, avoid module names as they may differ. It is fine if we get some false positives.
# - **IMPORTANT**: WHEN LAST INCOMPLETE LINE REFERENCES A FUNCTION/METHOD DIRECTLY, RETURN THAT NODE CIRRECTLY BY PAYING ATTENTION TO ANY ALIASES.
# - **IMPORTANT**: PAY ATTENTION TO THE PROVIDED GRAH SCHEMA TO MAKE CORRECT QUERIES.


# ## Input Data Format:
# Given repo_name: Repository name which can use to identify the Repo Node in the graph.
# Given file_name: File name which can use to identify the Module Node in the graph.
# Fetch the most important connected nodes from the graph to predict the next line of the below code:
# Incomplete code snippet to complete.

# ## Your Task:
# First provide a brief thought on your decision process, then generate **ONLY THE CYPHER QUERY**.

# **Format:**
# ```
# **Thought:** Incomplete element identified: <element_name> (function/method)
# **Query:**
# [Cypher query only]

# ## Example
# Given repo_name: /Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system
# Given file_name: models.classes
# Fetch the most important connected nodes from the graph to predict the next line of the below code:
# import numpy as np
# from poptransformer import ops
# from poptransformer.layers.layer_norm import BaseLayerNorm
# from classes import BaseModule as base_module

# class BaseRMSLayerNorm(BaseLayerNorm):
#     def __init__(self, input_size, eps=1e-5, context=''):
#         self.base_object = base_module()

#     def collect_bind_layer_weights(self):
#         weight_key = '.'.join([self.context, 'weight'])
#         weight_np = self.get_param_from_state_dict(weight_key, [self.input_size])
#         self.weight_id = self.add_initialized_input_tensor(weight_np, weight_key)

#     def __call__(self, graph, x):
#         variance_epsilon = ops.constant(graph, np.array(self.eps).astype(np.float32), 'variance_epsilon')
#         variance = self.base_object.

# **Thought:**  Incomplete method __call__ in BaseRMSLayerNorm class, remaining methods are not important. The last incomplete line uses self.base_object, which calls base_module but that is an ALIAS of the imported BaseModule class suggesting need for BaseModule and its methods. Also need parent class BaseLayerNorm methods for inheritance context.

# **Query:**
# ```cypher
# MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseRMSLayerNorm'}})-[:HAS_METHOD]->(method {{name: '__call__'}})-[:USES]->(dep)
# RETURN DISTINCT dep, labels(dep) as label
# UNION
# MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseModule'}})-[:HAS_METHOD]->(method)
# RETURN DISTINCT method as dep, labels(method) as label
# UNION
# MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseModule'}})
# RETURN DISTINCT c as dep, labels(c) as label
# UNION
# MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseLayerNorm'}})-[:HAS_METHOD]->(method)
# RETURN DISTINCT method as dep, labels(method) as label
# ```


# """

crosscode_query_generation_prompt_v3 = """
# Neo4j Cypher Query Expert for Code Dependency Analysis

You are a Neo4j Cypher query expert. Your task is to generate concise Cypher queries to find ALL cross file dependencies that will help to complete the provided incomplete code based on the provided graph schema.

## Graph Schema:
**Nodes**: Repo (name), Module, Class (name, code, signature, module_name), Function (name, code, signature, module_name), Method (name, code, signature, module_name, class)

**Edges**: CONTAINS (Repo→Module, Module→Class/Function), HAS_METHOD (Class→Method), INHERITS (Class→Class), USES (All→Dependencies)

#Instructions:
- **CORRECTNESS**: Use proper Cypher syntax. Ensure each UNION branch in Cypher has a complete MATCH...RETURN with SAME COLUMN NAMES.!
- **GENERATE MINIMAL QUERIES**:  ONLY RETIREVE THOSE NODES THAT YOU WILL REQUIRE TO COMPLETE THE INCOMPLETE CODE. Use fewest UNION clauses possible.
- **MANDATORY**: Return the enitre nodes as ***dep*** and their labels as ***label*** in the query. NOTE THE NAMES SHOULD BE 'dep' and 'label' ONLY.
- **IMPORTANT**: PAY EXTRA ATTENTION TO IMPORTS ALIASES, AND ONLY THE GLOBAL VARIABLES BEING USED IN THE LAST INCOMPLETE LINE.
- **IMPORTANT**: In the generated cypher query ONLY USE NAMES YOU ARE CONFIDENT ABOUT OR ELSE DON'T USE THEM.  For imports, avoid module names as they may differ. It is fine if we get some false positives.
- **IMPORTANT**: PAY ATTENTION TO THE PROVIDED GRAH SCHEMA TO MAKE CORRECT QUERIES. 
- **IMPORTANT**: When code references a function/method/class directly (e.g., `func_name()`), return that exact node too with its dependencies.


## Input Data Format:
Given repo_name: Repository name which can use to identify the Repo Node in the graph.
Given file_name: File name which can use to identify the Module Node in the graph.
Fetch the most important connected nodes from the graph to predict the next line of the below code:
Incomplete code snippet to complete.

## Your Task:
First provide a brief thought on your decision process, then generate **ONLY THE CYPHER QUERY**.

**Format:**
```
**Thought:** Incomplete element identified: <element_name> (function/method)
**Query:**
[Cypher query only]

#Example 
Given repo_name: /Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system
Given file_name: models.classes
Fetch the most important connected nodes from the graph to predict the next line of the below code:
import numpy as np
from poptransformer import ops
from poptransformer.layers.layer_norm import BaseLayerNorm
from classes import BaseModule as base_module

class BaseRMSLayerNorm(BaseLayerNorm):
    def __init__(self, input_size, eps=1e-5, context=''):    
        self.base_object = base_module()

    def collect_bind_layer_weights(self):
        weight_key = '.'.join([self.context, 'weight'])
        weight_np = self.get_param_from_state_dict(weight_key, [self.input_size])
        self.weight_id = self.add_initialized_input_tensor(weight_np, weight_key)

    def __call__(self, graph, x):
        variance_epsilon = ops.constant(graph, np.array(self.eps).astype(np.float32), 'variance_epsilon')
        variance = self.base_object.

**Thought:**  Incomplete method __call__ in BaseRMSLayerNorm class, remaining methods are not important. The last incomplete line uses self.base_object, which calls base_module but that is an ALIAS of the imported BaseModule class suggesting need for BaseModule . Also need parent class BaseLayerNorm  for inheritance context.

**Query:**
```cypher
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseRMSLayerNorm'}})-[:HAS_METHOD]->(method {{name: '__call__'}})-[:USES]->(dep)
RETURN DISTINCT dep, labels(dep) as label
UNION
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseModule'}})
RETURN DISTINCT c as dep, labels(c) as label
UNION
MATCH (r:Repo {{name: '/Users/pratik.shah1/work/CrossCodeEval_repos/google_alert-system'}})-[:CONTAINS]->(m:Module)-[:CONTAINS]->(c:Class {{name: 'BaseLayerNorm'}})
RETURN DISTINCT method as c, labels(c) as label
```


"""
