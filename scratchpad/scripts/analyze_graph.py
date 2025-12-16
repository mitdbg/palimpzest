import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import numpy as np
from palimpzest.core.data.graph_dataset import GraphDataset

GRAPH_FILE = Path("/Users/jason/projects/mit/palimpzest/data/cms_combined_graph.json")
OUTPUT_DIR = Path("/Users/jason/projects/mit/palimpzest/data/analysis_plots")

def analyze_graph():
    print(f"Loading graph from {GRAPH_FILE}...")
    graph = GraphDataset.load(GRAPH_FILE)
    
    nodes = list(graph._nodes_by_id.values())
    edges = list(graph._edges_by_id.values())
    
    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {len(edges)}")
    
    # --- Node Analysis ---
    
    # 1. Node Text Lengths (Stacked by Type)
    node_types_set = sorted(list(set(n.type for n in nodes if n.type)))
    text_lengths_by_type = []
    labels = []
    
    for nt in node_types_set:
        lengths = [len(n.text) if n.text else 0 for n in nodes if n.type == nt]
        text_lengths_by_type.append(lengths)
        labels.append(nt)
        
    # Handle nodes with no type if any
    no_type_lengths = [len(n.text) if n.text else 0 for n in nodes if not n.type]
    if no_type_lengths:
        text_lengths_by_type.append(no_type_lengths)
        labels.append("None")

    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths_by_type, bins=50, stacked=True, label=labels, edgecolor='black')
    plt.title('Distribution of Node Text Lengths by Type')
    plt.xlabel('Length (characters)')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(OUTPUT_DIR / "node_text_lengths.png")
    plt.close()
    
    all_lengths = [len(n.text) if n.text else 0 for n in nodes]
    print(f"Avg Text Length: {np.mean(all_lengths):.2f}")
    print(f"Max Text Length: {np.max(all_lengths)}")
    
    # 2. Node Types
    node_types = [n.type for n in nodes]
    type_counts = Counter(node_types)
    
    plt.figure(figsize=(10, 6))
    plt.bar(type_counts.keys(), type_counts.values(), color='lightgreen', edgecolor='black')
    plt.title('Distribution of Node Types')
    plt.xlabel('Node Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "node_types.png")
    plt.close()
    
    print("Node Types:", dict(type_counts))

    # --- Edge Analysis ---
    
    # 3. Edge Types
    edge_types = [e.type for e in edges]
    edge_type_counts = Counter(edge_types)
    
    plt.figure(figsize=(10, 6))
    plt.bar(edge_type_counts.keys(), edge_type_counts.values(), color='salmon', edgecolor='black')
    plt.title('Distribution of Edge Types')
    plt.xlabel('Edge Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "edge_types.png")
    plt.close()
    
    print("Edge Types:", dict(edge_type_counts))
    
    # 4. Node Degrees
    in_degrees = Counter()
    out_degrees = Counter()
    
    for edge in edges:
        out_degrees[edge.src] += 1
        in_degrees[edge.dst] += 1
        
    # Ensure all nodes are in the counters (even with 0 degree)
    for node in nodes:
        if node.id not in in_degrees:
            in_degrees[node.id] = 0
        if node.id not in out_degrees:
            out_degrees[node.id] = 0
            
    in_degree_vals = list(in_degrees.values())
    out_degree_vals = list(out_degrees.values())
    total_degree_vals = [in_degrees[n.id] + out_degrees[n.id] for n in nodes]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].hist(in_degree_vals, bins=30, color='orchid', edgecolor='black')
    axes[0].set_title('In-Degree Distribution')
    axes[0].set_xlabel('In-Degree')
    axes[0].set_ylabel('Count')
    axes[0].set_yscale('log')
    
    axes[1].hist(out_degree_vals, bins=30, color='orange', edgecolor='black')
    axes[1].set_title('Out-Degree Distribution')
    axes[1].set_xlabel('Out-Degree')
    axes[1].set_ylabel('Count')
    axes[1].set_yscale('log')

    axes[2].hist(total_degree_vals, bins=30, color='teal', edgecolor='black')
    axes[2].set_title('Total Degree Distribution')
    axes[2].set_xlabel('Total Degree')
    axes[2].set_ylabel('Count')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "node_degrees.png")
    plt.close()
    
    print(f"Avg In-Degree: {np.mean(in_degree_vals):.2f}")
    print(f"Max In-Degree: {np.max(in_degree_vals)}")
    print(f"Avg Out-Degree: {np.mean(out_degree_vals):.2f}")
    print(f"Max Out-Degree: {np.max(out_degree_vals)}")

if __name__ == "__main__":
    analyze_graph()
