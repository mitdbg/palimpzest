import os
import requests
import difflib
import base64
import ast
import logging
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler
import pygit2
import tiktoken

# TO DO: Generalize to other repos 
# TEMP_VARS = {
#     "owner": "astropy",
#     "repo": "astropy", 
#     "branch": "main", 
# }

# Rates per million tokens
MODEL_RATES = {
    "gpt-4o": {
        "input_rate": 2.5,
        "output_rate": 10, 
    },
}

def count_tokens(content: str) -> int: 
    encoding = tiktoken.encoding_for_model("gpt-4")  # or "gpt-3.5-turbo", etc.
    tokens = encoding.encode(content)
    num_tokens = len(tokens)
    return num_tokens


def add_line_numbers(code_str, start_line_no=1):
    """
    Takes a code string without line numbers and returns a JSON string
    where each line is represented as an object with its line number and content.
    """
    lines = code_str.splitlines()
    result = {str(i + 1 + (start_line_no - 1)): line for i, line in enumerate(lines)}
    return result

def fetch_github_code(file_name: str, owner: str, repo: str, base_commit: str = None, branch: str = "main") -> tuple[str, str]:
    """ 
    Fetches the code of a file returning the best match file path and its content. 
    """
    # Customize these variables for your repository.
    token = os.getenv("GITHUB_TOKEN")

    ref = base_commit if base_commit else branch
    file_paths = get_repo_files(owner, repo, ref, token)
    if file_paths is None:
        return None, None

    best_match = find_best_match(file_name, file_paths)
    if best_match:
        content = get_file_content(owner=owner, repo=repo, path=best_match, token=token, ref=ref)
        if content:
            return best_match, content

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

def get_file_content(owner, repo, path, token=None, ref="main"):
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
    
def search_keyword(repo_path, commit_hash, keyword):
    """
    Search a commit for files that contain the given keyword or its similar variations.
    
    The provided keyword is first split into tokens based on camel case, underscores,
    and periods. Candidate patterns are generated from contiguous token combinations.
    
    The file content is normalized (by removing underscores and periods and converting to lowercase)
    and then checked for the presence of any candidate string. This ensures that the tokens
    appear together (as a contiguous substring) rather than scattered throughout the file.
    
    Parameters:
        repo_path (str): Local path to the repository.
        commit_hash (str): Commit hash to search.
        keyword (str): The search keyword.
    
    Returns:
        list: File paths (relative to the repository root) that match.
    """
    
    repo = pygit2.Repository(repo_path)
    commit = repo[commit_hash]
    results = []

    # TO DO: Validate that this search is for the correct commit 
    # TO DO: Maybe implement a more efficient search
    def search_tree(tree, path_prefix=""):
        for entry in tree:
            full_path = f"{path_prefix}/{entry.name}" if path_prefix else entry.name
            if entry.type_str == 'blob':
                blob = repo[entry.id]
                try:
                    content = blob.data.decode('utf-8', errors='ignore')
                except Exception:
                    continue
                # A file is a match if any candidate appears as a contiguous substring.
                if keyword in content: 
                    results.append(full_path)
            elif entry.type_str == 'tree':
                search_tree(repo[entry.id], full_path)
    
    search_tree(commit.tree)

    return results

def download_repo(repo_name, dest_dir='repos'):
    """
    Download the GitHub repository if not already downloaded.
    
    Parameters:
        repo_name (str): Repository name in the format "owner/repo"
        dest_dir (str): Directory to store repositories (default: 'repos')
    
    Returns:
        str: Local path of the repository.
    """

    owner, repo = repo_name.split('/')
    repo_url = f"https://github.com/{owner}/{repo}.git"
    local_path = os.path.join(dest_dir, repo)

    if not os.path.isdir(local_path):
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Cloning '{repo_url}' into '{local_path}'...")
        pygit2.clone_repository(repo_url, local_path)
    
    return local_path

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

def compute_cost_from_history(history, model='gpt-4o'):
    total_cost = 0
    for prompt in history: 
        output_tokens = prompt['usage']['completion_tokens']
        input_tokens = prompt['usage']['prompt_tokens']
        input_rate = MODEL_RATES[model]['input_rate']
        output_rate = MODEL_RATES[model]['output_rate']
        total_cost += (output_tokens / 10**6) * output_rate + (input_tokens / 10**6) * input_rate
    return total_cost 

# Configure logging
def setup_logger(log_dir="logs", max_bytes=5_000_000, backup_count=10):
    """
    Sets up a rotating file logger.

    Args:
        log_dir (str): Directory where log files will be stored.
        log_file (str): Name of the log file.
        max_bytes (int): Maximum file size before rotating.
        backup_count (int): Number of backup log files to keep.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create full log file path
    log_file = f'app_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    log_path = os.path.join(log_dir, log_file)
    
    # Set up logging
    logger = logging.getLogger("AppLogger")
    logger.setLevel(logging.INFO)
    
    # Create rotating file handler
    handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add the handler to the logger
    logger.addHandler(handler)

    # Prevent logs from propagating up to the root logger
    logger.propagate = False
    
    return logger

def add_patch_to_output_dir(file_path, new_data, indent=4):
    """ Adds patch content to output json file to save incremental progress """
    
    # Ensure the file exists, if not, create it with an empty JSON list
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w") as file:
            json.dump([], file, indent=indent)

    # Open file in read+write binary mode
    with open(file_path, "rb+") as file:
        file.seek(-1, os.SEEK_END)  # Move to the last character

        while file.tell() > 0:
            char = file.read(1).decode("utf-8")
            if char in "]":  # Find the closing bracket
                file.seek(-1, os.SEEK_CUR)  # Move back one step
                break
            file.seek(-2, os.SEEK_CUR)  # Move back if it's a space or newline
        
        if file.tell() > 1:  # If there's already data in the list
            file.write(b",\n")  # Add a comma and new line before appending
        else: 
            file.write(b"\n")

        # Append new data with proper indentation
        formatted_entry = json.dumps(new_data, indent=indent)
        formatted_entry = "\n".join([" " * indent + line for line in formatted_entry.splitlines()])  # Indent each line

        file.write(formatted_entry.encode("utf-8"))  # Write new entry
        file.write(b"\n]")  # Close the JSON list

# ------- Patch Data Utilities -------

def verify_and_fix_patch_data(patch_data: dict, instance_params: dict, instance_id: str):
    """
    Expects patch data in the following format:

    {
        "old_path": "old/file.txt",
        "new_path": "new/file.txt",
        "hunks": [
        {
            "old_start": 1,
            "old_length": 3,
            "new_start": 1,
            "new_length": 3,
            "hunk_header_context": "def my_function():",
            "lines": [
                {"type": "context", "content": "unchanged line"},
                {"type": "addition", "content": "added line"},
                {"type": "deletion", "content": "removed line"}
            ]
        }
    }

    - Verifies the structure of the patch data
    - Corrects any incorrect old and new line counts for each hunk
    """

    if "hunks" not in patch_data:
        raise KeyError("Missing 'hunks' key in file data")
    
    # Test that the file exists
    try: 
        file_path = patch_data["old_path"]
        base_commit = instance_params[instance_id]["base_commit"]
        owner = instance_params[instance_id]["owner"]
        repo = instance_params[instance_id]["repo"]
        best_match, _ = fetch_github_code(file_path, owner, repo, base_commit)
        if best_match != file_path:
            return f"File {file_path} not found in the repo. Did you mean {best_match}?"
    except KeyError as e:
        raise KeyError(f"Invalid instance_id")

    for hunk in patch_data["hunks"]: 
        
        # Check hunk keys
        if "old_length" not in hunk or "new_length" not in hunk:
            raise KeyError("Missing 'old_length' or 'new_length' key in hunk data")
        if hunk["old_start"] == "XXX":
            raise ValueError(f"Missing line numbers in patch data for {patch_data['old_path']}")

        num_old_lines, num_new_lines = 0, 0

        for line in hunk.get("lines"):
            # Check line keys
            if "type" not in line:
                raise KeyError(f"Missing 'type' key in line data for {patch_data['old_path']}")
            if "content" not in line:
                raise KeyError(f"Missing 'content' key in line data for {patch_data['old_path']}")

            if line["type"] == "context":
                num_old_lines += 1
                num_new_lines += 1
            elif line["type"] == "addition":
                num_new_lines += 1
            elif line["type"] == "deletion":
                num_old_lines += 1
        
        # Fix any incorrect counts
        if num_old_lines != hunk["old_length"]:
            hunk["old_length"] = num_old_lines
        if num_new_lines != hunk["new_length"]:
            hunk["new_length"] = num_new_lines

def minimize_patch_data(patch_data):
    """
    Given a patch_data dictionary, mutates it in-place to contain only minimal hunks.
    Keeps one context line before and/or after changes if available.
    """
    for hunk in patch_data["hunks"]:
        lines = hunk["lines"]

        # Find indices of the first and last changed lines
        change_indices = [i for i, line in enumerate(lines) if line["type"] in ("addition", "deletion")]
        if not change_indices:
            continue  # Nothing to minimize if no changes

        start = max(0, change_indices[0] - 1)  # one line before
        end = min(len(lines), change_indices[-1] + 2)  # one line after

        # Slice to minimal lines
        minimal_lines = lines[start:end]
        hunk["lines"] = minimal_lines

        # Recalculate hunk lengths
        old_count = sum(1 for line in minimal_lines if line["type"] != "addition")
        new_count = sum(1 for line in minimal_lines if line["type"] != "deletion")
        hunk["old_length"] = old_count
        hunk["new_length"] = new_count

        # Adjust start lines
        context_before = sum(1 for i in range(start) if lines[i]["type"] != "addition")
        hunk["old_start"] += context_before
        hunk["new_start"] += context_before

    return patch_data

def write_target_file(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
