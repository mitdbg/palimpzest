import json
from pathlib import Path
from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

def standardize_schema():
    print(f"Loading graph from {GRAPH_FILE}...")
    graph = GraphDataset.load(GRAPH_FILE)
    
    print("Standardizing schema...")
    
    for node in graph._nodes_by_id.values():
        # --- git_docs ---
        if node.type == "git_docs":
            # Set name
            if "title" in node.attrs:
                node.attrs["name"] = node.attrs["title"]
            else:
                node.attrs["name"] = node.attrs.get("display_name", "Untitled")
            
            # Set level
            node.attrs["level"] = 0
            
        # --- git_directory ---
        elif node.type == "git_directory":
            # Set name
            path_str = node.attrs.get("path", "")
            if path_str == "git_root":
                node.attrs["name"] = "Git Root"
            else:
                node.attrs["name"] = Path(path_str).name
            
            # Set level (Directories group files, so Level 1)
            node.attrs["level"] = 1
            
        # --- jira_tickets ---
        elif node.type == "jira_tickets":
            meta = node.attrs.get("metadata", {})
            
            # Set name
            if "name" in meta:
                node.attrs["name"] = meta["name"]
            
            # Set level based on original_level
            orig_level = meta.get("original_level")
            if orig_level == "ticket":
                node.attrs["level"] = 0
            elif orig_level == "cluster":
                node.attrs["level"] = 1
            elif orig_level == "project":
                node.attrs["level"] = 2
            elif orig_level == "root":
                node.attrs["level"] = 3
            else:
                # Fallback if original_level is missing, though analysis showed 100% coverage
                # Use existing level if it maps correctly? 
                # Previous analysis: Root=3, Project=2, Cluster=1. 
                # So existing level might actually be correct for non-tickets.
                # But let's stick to the mapping for consistency.
                pass
            
            # Remove metadata dict
            node.attrs.pop("metadata", None)

    print("Schema standardization complete.")
    
    # Verification step
    print("\nVerifying Schema:")
    nodes_by_type = {}
    for node in graph._nodes_by_id.values():
        if node.type not in nodes_by_type:
            nodes_by_type[node.type] = []
        nodes_by_type[node.type].append(node)
        
    for node_type, nodes in nodes_by_type.items():
        print(f"\nType: {node_type}")
        sample = nodes[0]
        print(f"  Sample Name: {sample.attrs.get('name')}")
        print(f"  Sample Level: {sample.attrs.get('level')}")
        print(f"  Has Metadata? {'metadata' in sample.attrs}")

    graph.save(GRAPH_FILE)
    print(f"\nSaved standardized graph to {GRAPH_FILE}")

if __name__ == "__main__":
    standardize_schema()
