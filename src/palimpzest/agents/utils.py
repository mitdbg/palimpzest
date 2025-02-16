import os
import requests
import difflib
import base64
import ast

def fetch_github_code(file_name: str, base_commit: str = None) -> str:
    """ Fetches the code of a file from the relevant issue code """
    # Customize these variables for your repository.
    owner = "astropy"
    repo = "astropy"
    branch = "main" 
    token = os.getenv("GITHUB_TOKEN")

    print("Fetching repository file list...")
    ref = base_commit if base_commit else branch 
    file_paths = get_repo_files(owner, repo, ref, token)
    if file_paths is None:
        return

    best_match = find_best_match(file_name, file_paths)
    if best_match:
        print(f"\nBest match found: {best_match}\n")
        content = get_file_content(owner, repo, best_match, branch, token)
        if content:
            return content

def get_repo_files(owner, repo, ref="main", token=None):
    """
    Fetch the full file tree for a given repository and ref.
    Uses GitHub's Git Trees API with the recursive option.
    """

    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Return only files (type "blob")
        file_paths = [item["path"] for item in data.get("tree", []) if item["type"] == "blob"]
        return file_paths
    else:
        print("Error fetching repository file tree:", response.status_code, response.text)
        return None

def find_best_match(query, file_paths):
    """
    Use fuzzy matching to select the file path that best matches the query.
    
    First, attempt a direct fuzzy match against the full paths.
    If no good match is found and the query includes directories,
    fall back to matching just the basename.
    """
    
    # Try matching against full paths.
    matches = difflib.get_close_matches(query, file_paths, n=1, cutoff=0.1)
    if matches:
        return matches[0]
    
    # Fallback: compare basenames.
    base_query = os.path.basename(query)
    best_match = None
    best_ratio = 0
    for path in file_paths:
        ratio = difflib.SequenceMatcher(None, base_query, os.path.basename(path)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = path
    # Adjust the threshold as needed.
    if best_ratio > 0.5:
        return best_match
    return None

def get_file_content(owner, repo, path, ref="main", token=None):
    """
    Fetch the file content from the GitHub Contents API.
    The returned content is Base64-encoded if it's a text file.
    """

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data.get("encoding") == "base64":
            try:
                decoded_bytes = base64.b64decode(data.get("content", ""))
                return decoded_bytes.decode('utf-8')
            except Exception as e:
                print("Error decoding file content:", e)
                return None
        else:
            return data.get("content")
    else:
        print("Error fetching file content:", response.status_code, response.text)
        return None

class FunctionClassVisitor(ast.NodeVisitor):
    def __init__(self):
        # Structure to store the results:
        # "classes" maps class names to a list of methods.
        # "functions" is a list of top-level function names.
        self.structure = {"classes": {}, "functions": []}
        # Stack to keep track of nested classes, if any.
        self.class_stack = []

    def visit_ClassDef(self, node):
        # Push the current class name onto the stack.
        self.class_stack.append(node.name)
        # Initialize the methods list for this class.
        self.structure["classes"][node.name] = []
        # Process the body of the class.
        self.generic_visit(node)
        # Pop when done.
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        if self.class_stack:
            # If we're inside a class, add this function as a method.
            current_class = self.class_stack[-1]
            self.structure["classes"][current_class].append(node.name)
        else:
            # Otherwise, it's a standalone (top-level) function.
            self.structure["functions"].append(node.name)
        # Continue processing any nested definitions.
        self.generic_visit(node)

def extract_structure(code):
    """
    Given source code as a string, returns a dictionary with:
      - 'functions': a list of standalone function names.
      - 'classes': a dict mapping class names to lists of method names.
    """
    tree = ast.parse(code)
    visitor = FunctionClassVisitor()
    visitor.visit(tree)
    return visitor.structure