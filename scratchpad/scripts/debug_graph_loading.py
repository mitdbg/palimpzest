import json
from pathlib import Path
from palimpzest.core.data.graph_dataset import GraphDataset, GraphSnapshot

def main():
    # Try to load cms_v1_graph.json
    path = Path("data/cms_v1_graph.json")
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    try:
        with open(path) as f:
            data = json.load(f)
        
        # Check first node in JSON
        if data['nodes']:
            print(f"JSON Node 0 type: {data['nodes'][0].get('type')}")
        
        snapshot = GraphSnapshot.model_validate(data)
        graph = GraphDataset.from_snapshot(snapshot)
        
        # Check first node in GraphDataset
        nodes = list(graph.store.iter_nodes())
        if nodes:
            print(f"GraphNode 0 type: {nodes[0].type}")
            
        # Count types
        types = {}
        for n in nodes:
            t = n.type or "None"
            types[t] = types.get(t, 0) + 1
            
        print("Node types distribution:")
        for t, count in types.items():
            print(f"  {t}: {count}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
