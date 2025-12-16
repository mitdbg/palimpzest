import json
from pathlib import Path
from collections import Counter
from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

def inspect_schemas():
    print(f"Loading graph from {GRAPH_FILE}...")
    graph = GraphDataset.load(GRAPH_FILE)
    
    nodes_by_type = {}
    for node in graph._nodes_by_id.values():
        if node.type not in nodes_by_type:
            nodes_by_type[node.type] = []
        nodes_by_type[node.type].append(node)
        
    for node_type, nodes in nodes_by_type.items():
        print(f"\n=== Node Type: {node_type} (Count: {len(nodes)}) ===")
        
        # Collect all attribute keys
        all_keys = Counter()
        for node in nodes:
            all_keys.update(node.attrs.keys())
            
        print("Attribute Keys Frequency:")
        for key, count in all_keys.most_common(20):
            print(f"  - {key}: {count} ({count/len(nodes)*100:.1f}%)")
            
        # Sample a few nodes to see values
        print("\nSample Node Attributes:")
        for i, node in enumerate(nodes[:3]):
            print(f"  Node {i+1} (ID: {node.id}):")
            # Print a summary of attrs
            for k, v in node.attrs.items():
                val_str = str(v)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"    {k}: {val_str}")
            print("-" * 20)

        # Inspect metadata for jira_tickets
        if node_type == "jira_tickets":
            print("\n--- jira_tickets Metadata Analysis ---")
            metadata_keys = Counter()
            original_levels = Counter()
            
            for node in nodes:
                meta = node.attrs.get("metadata", {})
                if isinstance(meta, dict):
                    metadata_keys.update(meta.keys())
                    if "original_level" in meta:
                        original_levels[meta["original_level"]] += 1
            
            print("Metadata Keys:")
            for key, count in metadata_keys.most_common(20):
                print(f"  - {key}: {count}")
                
            print("\nOriginal Levels:")
            for level, count in original_levels.most_common(20):
                print(f"  - {level}: {count}")

if __name__ == "__main__":
    inspect_schemas()
