import sys
from pathlib import Path
from palimpzest.core.data.graph_dataset import GraphDataset

GIT_SOURCES_GRAPH = Path("/Users/jason/projects/mit/palimpzest/data/git_sources_graph.json")
CMS_STANDARD_GRAPH = Path("/Users/jason/projects/mit/palimpzest/CURRENT_WORKSTREAM/exports/cms_standard_graph_snapshot.json")
OUTPUT_GRAPH = Path("/Users/jason/projects/mit/palimpzest/data/cms_combined_graph.json")

def union_graphs():
    print(f"Loading git sources graph from {GIT_SOURCES_GRAPH}...")
    if not GIT_SOURCES_GRAPH.exists():
        print(f"Error: {GIT_SOURCES_GRAPH} does not exist.")
        return

    graph1 = GraphDataset.load(GIT_SOURCES_GRAPH)
    print(f"Graph 1 has {len(graph1._nodes_by_id)} nodes and {len(graph1._edges_by_id)} edges.")

    print(f"Loading CMS standard graph from {CMS_STANDARD_GRAPH}...")
    if not CMS_STANDARD_GRAPH.exists():
        print(f"Error: {CMS_STANDARD_GRAPH} does not exist.")
        return

    graph2 = GraphDataset.load(CMS_STANDARD_GRAPH)
    print(f"Graph 2 has {len(graph2._nodes_by_id)} nodes and {len(graph2._edges_by_id)} edges.")

    print("Merging Graph 2 into Graph 1...")
    
    # Merge nodes
    nodes_added = 0
    nodes_skipped = 0
    for node in graph2._nodes_by_id.values():
        try:
            # We use overwrite=False to preserve existing nodes if IDs collide, 
            # or we could use overwrite=True if we prefer the second graph's version.
            # Given the user asked to "union", usually we want to keep existing or merge.
            # Let's assume we want to add missing nodes.
            if node.id not in graph1._nodes_by_id:
                graph1.add_node(node)
                nodes_added += 1
            else:
                nodes_skipped += 1
        except Exception as e:
            print(f"Error adding node {node.id}: {e}")

    # Merge edges
    edges_added = 0
    edges_skipped = 0
    for edge in graph2._edges_by_id.values():
        try:
            if edge.id not in graph1._edges_by_id:
                # Ensure src and dst exist in graph1 (they should if we added all nodes)
                if edge.src in graph1._nodes_by_id and edge.dst in graph1._nodes_by_id:
                    graph1.add_edge(edge)
                    edges_added += 1
                else:
                    print(f"Skipping edge {edge.id} because src/dst missing in target graph.")
            else:
                edges_skipped += 1
        except Exception as e:
            print(f"Error adding edge {edge.id}: {e}")

    print(f"Merge complete.")
    print(f"Nodes added: {nodes_added}, Skipped (duplicate ID): {nodes_skipped}")
    print(f"Edges added: {edges_added}, Skipped (duplicate ID): {edges_skipped}")
    print(f"Resulting Graph has {len(graph1._nodes_by_id)} nodes and {len(graph1._edges_by_id)} edges.")

    print(f"Saving combined graph to {OUTPUT_GRAPH}...")
    graph1.save(OUTPUT_GRAPH)
    print("Done.")

if __name__ == "__main__":
    union_graphs()
