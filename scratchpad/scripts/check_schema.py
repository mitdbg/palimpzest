import json
from pathlib import Path
from collections import Counter
from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

def check_schema_consistency():
    print(f"Loading graph from {GRAPH_FILE}...")
    graph = GraphDataset.load(GRAPH_FILE)
    
    all_keys = set()
    keys_by_type = {}
    
    for node in graph._nodes_by_id.values():
        keys = frozenset(node.attrs.keys())
        all_keys.update(keys)
        
        if node.type not in keys_by_type:
            keys_by_type[node.type] = Counter()
        keys_by_type[node.type][keys] += 1

    print("\n--- Schema Consistency Check ---")
    for node_type, counts in keys_by_type.items():
        print(f"\nType: {node_type}")
        for keys, count in counts.items():
            print(f"  Keys: {sorted(list(keys))} -> Count: {count}")
            
    print("\n--- Extra Attributes ---")
    # Check if any nodes have extra attributes besides 'name' and 'level'
    # (and the git specific ones we might have left)
    
    for node in graph._nodes_by_id.values():
        extra_keys = set(node.attrs.keys()) - {"name", "level"}
        if extra_keys:
            # Just print a few examples per type
            print(f"Node {node.id} ({node.type}) has extra keys: {extra_keys}")
            break # Just one example to see what's left

if __name__ == "__main__":
    check_schema_consistency()
