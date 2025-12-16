import json
from pathlib import Path
from collections import Counter
from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

def enforce_schema():
    print(f"Loading graph from {GRAPH_FILE}...")
    graph = GraphDataset.load(GRAPH_FILE)
    
    print("Enforcing schema structure...")
    
    for node in graph._nodes_by_id.values():
        # 1. Base Schema Enforcement
        # Ensure 'name' and 'level' exist
        if "name" not in node.attrs:
            node.attrs["name"] = node.label or "Untitled"
        if "level" not in node.attrs:
            node.attrs["level"] = 0 # Default to leaf
            
        # Add 'source' attribute to Base Schema
        if node.type == "jira_tickets":
            node.attrs["source"] = "jira"
        elif node.type.startswith("git_"):
            node.attrs["source"] = "git"
        else:
            node.attrs["source"] = "unknown"

        # 2. Type-Specific Cleanup (Extensions)
        if node.type == "git_docs":
            # Extension: path, url
            # Remove redundant keys
            keys_to_remove = ["display_name", "title", "source_type", "suffix"]
            for k in keys_to_remove:
                node.attrs.pop(k, None)
                
        elif node.type == "git_directory":
            # Extension: path
            pass
            
        elif node.type == "jira_tickets":
            # Extension: None currently
            pass

    graph.save(GRAPH_FILE)
    print("Schema enforcement complete.")
    
    # Verification
    print("\n--- Final Schema Verification ---")
    keys_by_type = {}
    for node in graph._nodes_by_id.values():
        keys = frozenset(node.attrs.keys())
        if node.type not in keys_by_type:
            keys_by_type[node.type] = Counter()
        keys_by_type[node.type][keys] += 1
        
    for node_type, counts in keys_by_type.items():
        print(f"\nType: {node_type}")
        for keys, count in counts.items():
            print(f"  Keys: {sorted(list(keys))} -> Count: {count}")

if __name__ == "__main__":
    enforce_schema()
