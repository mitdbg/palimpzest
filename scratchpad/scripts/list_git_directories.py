import json
from pathlib import Path
from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

def list_git_directories():
    print(f"Loading graph from {GRAPH_FILE}...")
    graph = GraphDataset.load(GRAPH_FILE)
    
    git_dirs = [n for n in graph._nodes_by_id.values() if n.type == "git_directory"]
    
    print(f"Found {len(git_dirs)} git_directory nodes:")
    
    # Sort by path for better readability
    git_dirs.sort(key=lambda n: n.attrs.get("path", ""))
    
    for node in git_dirs:
        print(f"ID: {node.id}")
        print(f"  Path: {node.attrs.get('path')}")
        print(f"  Label: {node.label}")
        print("-" * 20)

if __name__ == "__main__":
    list_git_directories()
