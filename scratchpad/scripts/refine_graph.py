import json
from pathlib import Path
from palimpzest.core.data.graph_dataset import GraphDataset, GraphNode, GraphEdge

INPUT_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_combined_graph.json")
OUTPUT_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_refined_graph.json")

def refine_graph():
    print(f"Loading graph from {INPUT_FILE}...")
    graph = GraphDataset.load(INPUT_FILE)
    
    # 1. Rename types and remove attributes
    print("Refining node types and attributes...")
    for node in graph._nodes_by_id.values():
        if node.type == "cms_block":
            node.type = "jira_tickets"
            node.attrs.pop("relevance", None)
            node.attrs.pop("access_count", None)
        elif node.type == "document":
            node.type = "git_docs"
            node.attrs.pop("suffix", None)
            
    # 2. Extract hierarchy from git_docs
    print("Extracting git hierarchy...")
    
    dir_nodes = {} # path_str -> node_id
    
    def get_or_create_dir(dir_path_str, label):
        if dir_path_str in dir_nodes:
            return dir_nodes[dir_path_str]
        
        node_id = f"dir:{dir_path_str}"
        
        if node_id not in graph._nodes_by_id:
            node = GraphNode(
                id=node_id,
                type="git_directory",
                text=label,
                label=label,
                attrs={"path": dir_path_str}
            )
            graph.add_node(node)
            
        dir_nodes[dir_path_str] = node_id
        return node_id

    # Create a root node for all git docs
    root_id = get_or_create_dir("git_root", "Git Root")

    git_docs = [n for n in graph._nodes_by_id.values() if n.type == "git_docs"]
    
    for node in git_docs:
        path_str = node.attrs.get("path")
        if not path_str:
            continue
            
        path = Path(path_str)
        
        # Determine parent directory
        if len(path.parts) > 1:
            parent_dir = path.parent
            parent_dir_str = str(parent_dir)
            parent_id = get_or_create_dir(parent_dir_str, parent_dir.name)
        else:
            # File is at root
            parent_id = root_id
            
        # Link file to parent
        edge_id = f"{parent_id}->{node.id}"
        if edge_id not in graph._edges_by_id:
            edge = GraphEdge(
                id=edge_id,
                src=parent_id,
                dst=node.id,
                type="hierarchy:child"
            )
            graph.add_edge(edge)
            
        # Walk up the directory tree to link directories
        if len(path.parts) > 1:
            current_dir = path.parent
            while str(current_dir) != ".":
                current_dir_str = str(current_dir)
                current_id = get_or_create_dir(current_dir_str, current_dir.name)
                
                if len(current_dir.parts) > 1:
                    parent_of_current = current_dir.parent
                    parent_of_current_str = str(parent_of_current)
                    parent_of_current_id = get_or_create_dir(parent_of_current_str, parent_of_current.name)
                else:
                    parent_of_current_id = root_id
                
                # Link directory to parent directory
                edge_id = f"{parent_of_current_id}->{current_id}"
                if edge_id not in graph._edges_by_id:
                    edge = GraphEdge(
                        id=edge_id,
                        src=parent_of_current_id,
                        dst=current_id,
                        type="hierarchy:child"
                    )
                    graph.add_edge(edge)
                
                if len(current_dir.parts) == 1:
                    break
                current_dir = current_dir.parent

    print(f"Added {len(dir_nodes)} directory nodes.")
    print(f"Total Nodes: {len(graph._nodes_by_id)}")
    print(f"Total Edges: {len(graph._edges_by_id)}")
    
    graph.save(OUTPUT_FILE)
    print(f"Saved refined graph to {OUTPUT_FILE}")

if __name__ == "__main__":
    refine_graph()
