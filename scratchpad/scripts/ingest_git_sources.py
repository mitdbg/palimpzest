import os
import yaml
from pathlib import Path
from palimpzest.core.data.graph_dataset import GraphDataset, GraphNode

DATA_DIR = Path("/Users/jason/projects/mit/palimpzest/data/git_sources")
OUTPUT_FILE = Path("/Users/jason/projects/mit/palimpzest/data/git_sources_graph.json")

def ingest():
    print(f"Ingesting from {DATA_DIR}...")
    graph = GraphDataset(name="git_sources")
    
    # List all .txt files
    files = list(DATA_DIR.glob("*.txt"))
    print(f"Found {len(files)} text files.")
    
    for txt_file in files:
        meta_file = txt_file.with_suffix(".txt.meta.yaml")
        
        if not meta_file.exists():
            print(f"Warning: No metadata for {txt_file}")
            continue
            
        # Read content
        try:
            content = txt_file.read_text(errors="replace")
        except Exception as e:
            print(f"Error reading content for {txt_file}: {e}")
            continue
        
        # Read metadata
        try:
            meta = yaml.safe_load(meta_file.read_text())
        except Exception as e:
            print(f"Error reading metadata for {txt_file}: {e}")
            continue
            
        # Create node
        # Use filename stem as ID (e.g. 100405622916)
        node_id = txt_file.stem
        
        node = GraphNode(
            id=node_id,
            text=content,
            attrs=meta,
            type="document",
            label=meta.get("title", "Untitled")
        )
        
        graph.add_node(node)
        
    print(f"Created graph with {len(graph._nodes_by_id)} nodes.")
    
    # Save
    graph.save(OUTPUT_FILE)
    print(f"Saved graph to {OUTPUT_FILE}")

if __name__ == "__main__":
    ingest()
