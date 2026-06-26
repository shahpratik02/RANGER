summarisation_prompt = """
### Task: Code Summarization

Summarize the code at a high level without referencing specific function or variable names. Focus on its purpose, how it is implemented, and its notable features. Use the following format:

**PURPOSE**
Describe what the code is designed to achieve.

**IMPLEMENTATION**
Explain how the code accomplishes its purpose, including general techniques or components used, without naming exact functions or variables.

**KEY FEATURES**
List significant capabilities, design patterns, or behaviors the code exhibits.

### Programming Language: Python
### Code:

"""

members_prompt = """
### Task: Code Members Description

Analyze the Python code and identify important variables(skip temporary variables and trivial assignments), functions and classes (also function calls and class instantiations). Use the following format:

name - description

List each important code member with its name followed by a dash and a *** one-line shortdescription *** of its purpose or functionality.

If no important members are found, respond with: ---None---

***DO NOT REPEAT MEMBERS. YOU CAN CONCLUDE EARLY ONCE ALL MEMBERS ARE LISTED.***


### Programming Language: Python
### Code:

"""


file_prompt_2 = """
### Task: File Summary from Member Descriptions

Create a high-level summary of a Python file based on the provided member descriptions. You are not given any code, but only the descriptions of parts of the code given by various developers.
 You have to use ALL these descriptions to summarize the code.

### Guidelines:
1. Do not include any code in your response, or guess the code. Simply try and summarize the descriptions provided to you.
2. Focus on the file's overall purpose, architecture, key functionality, and key members.
3. If no description is provided simply say 'No description found'.
4. Summarize the purpose of ALL components mentioned in the descriptions.


"""

file_prompt = """
### Task: Create a Summary Report
You are a technical documentation specialist. Read the following function descriptions and create a concise summary report.

**Important:** 
Go thorugh each description and then create a DETIALED CONCISE summary of each description and then concatenate them.
In your summaries INCLUDE ALL variable names, function names, class names, etc. from the descriptions.
DO NOT REPEAT LINES IN OUTPUT

### Function Descriptions:


"""