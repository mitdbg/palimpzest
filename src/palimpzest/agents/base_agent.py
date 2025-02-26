
import palimpzest.agents.utils as utils
from palimpzest.query.generators.generators import get_api_key
import dspy
import ast
import re

LOGGER = utils.setup_logger()

openai_key = get_api_key("OPENAI_API_KEY")

# GLOBAL VARIABLES
# TO DO: Find a better way to create state accessible  
GLOBAL_CONTEXT = {
    "relevant_issue_code": None,
    "model": dspy.LM('openai/gpt-4o', api_key=openai_key),
    "base_commit": None,
}

class BaseAgent:

    @staticmethod
    def set_globals(relevant_issue_code: str, base_commit: str):
        """
        Sets the global variables for the agent.
        """
        GLOBAL_CONTEXT["relevant_issue_code"] = relevant_issue_code
        GLOBAL_CONTEXT["base_commit"] = base_commit

    @staticmethod
    def get_file_content(file_name: str, include_line_numbers: bool) -> str:
        """
        Returns the content of an entire file. 
        Use this when the entire context of a file is imporant.
        Only set include_line_numbers to True if being used to generate a patch. 
        """
        content = utils.fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
        return utils.add_line_numbers(content) if include_line_numbers else content

    @staticmethod
    def search_keyword(keywords: list[str]) -> str:
        """
        Searches the codebase for keywords and returns the files that contain them.
        """
        results = utils.search_files(keywords)

        if results and "items" in results:
            print(f"Found {results['total_count']} matching files:")
            relevant_files = ', '.join([item['name'] for item in results["items"]])
            return relevant_files 
        else:
            print("No results found.")
            return "No Results found"

    @staticmethod
    def extract_method(file_name: str, function_name: str, include_line_numbers: bool=False) -> str: 
        """
        Extracts the implementation of a function from a file. 
        Use this when you only need a single function in the file.
        Only set include_line_numbers to True if being used to generate a patch.
        """

        try:
            # Parse the source code into an AST
            content = utils.fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
            tree = ast.parse(content)
        except SyntaxError as e:
            return "extract_method(): Error parsing the code"

        # Walk through all nodes in the AST.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_line = node.lineno
                end_line = node.end_lineno

                # Split the content into lines and extract the block.
                lines = content.splitlines()
                function_lines = lines[start_line - 1:end_line]
                method_str = "\n".join(function_lines)

                if include_line_numbers: 
                    return utils.add_line_numbers(method_str, start_line_no=start_line)
                else:
                    return method_str

        return "function not found in the file"

    @staticmethod
    def get_classes_and_methods(file_name: str) -> str:
        """ 
        Summarizes all the classes and methods in a file, returning a dict where keys are classes and values are a list of method. 
        Useful for understanding the structure of a file for subsequent method extraction. 
        """

        relevant_issue_code = GLOBAL_CONTEXT["relevant_issue_code"]

        pattern = rf"\[start of ([^\]]*{re.escape(file_name)}[^\]]*)\](.*?)\[end of \1\]"
        match = re.search(pattern, relevant_issue_code, re.DOTALL)
        
        if match: 
            code = match.group(2).strip()
        else: 
            code = utils.fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
            if not code: 
                return "That file is not found in the relevant issue code, please try another file"

        code_structure = utils.extract_structure(code)
        return code_structure 
    

# Add optional line number args, and add them to the output