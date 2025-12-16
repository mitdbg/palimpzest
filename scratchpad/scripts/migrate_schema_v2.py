import json
from pathlib import Path
from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

def migrate_schema():
    print(f"Loading graph from {GRAPH_FILE}...")
    # This might fail if Pydantic is very strict, but fields are optional so it should be fine.
    graph = GraphDataset.load(GRAPH_FILE)
    
    print("Migrating schema...")
    
    count_level = 0
    count_source = 0
    count_name_removed = 0
    
    for node in graph._nodes_by_id.values():
        # Move level
        if "level" in node.attrs:
            node.level = node.attrs.pop("level")
            count_level += 1
            
        # Move source
        if "source" in node.attrs:
            node.source = node.attrs.pop("source")
            count_source += 1
            
        # Remove name (redundant with label)
        if "name" in node.attrs:
            # Ensure label is set if it wasn't (though it should be)
            if not node.label:
                node.label = node.attrs["name"]
            node.attrs.pop("name")
            count_name_removed += 1

    print(f"Moved 'level' for {count_level} nodes.")
    print(f"Moved 'source' for {count_source} nodes.")
    print(f"Removed 'name' from {count_name_removed} nodes.")
    
    graph.save(GRAPH_FILE)
    print(f"Saved migrated graph to {GRAPH_FILE}")

    # Verification
    print("\n--- Verification ---")
    sample_nodes = list(graph._nodes_by_id.values())[:3]
    for node in sample_nodes:
        print(f"ID: {node.id}")
        print(f"  Type: {node.type}")
        print(f"  Label: {node.label}")
        print(f"  Level: {node.level} (Top-level)")
        print(f"  Source: {node.source} (Top-level)")
        print(f"  Attrs: {node.attrs}")
        print("-" * 20)

if __name__ == "__main__":
    migrate_schema()
