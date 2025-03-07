#!/usr/bin/env python3
"""
Generate sample metrics for the knowledge graph.
"""

import os
import json
import networkx as nx
import datetime
import random

def generate_metrics():
    """
    Generate sample metrics for the knowledge graph.
    """
    # Create output directory if it doesn't exist
    os.makedirs('pipeline_demo_output/graph_analysis', exist_ok=True)
    
    # Load the graph
    try:
        G = nx.read_gexf('pipeline_demo_output/final_knowledge_graph.gexf')
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    except Exception as e:
        print(f"Error loading graph: {e}")
        # Create a sample graph
        G = nx.DiGraph()
        for i in range(100):
            G.add_node(f"node_{i}", type="persona" if i < 50 else "concept")
        
        for i in range(200):
            source = f"node_{random.randint(0, 99)}"
            target = f"node_{random.randint(0, 99)}"
            if source != target:
                G.add_edge(source, target, weight=random.random())
        
        print(f"Created sample graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate metrics for 5 iterations
    for iteration in range(5):
        # Calculate basic metrics
        num_nodes = G.number_of_nodes() - random.randint(0, 10) * (4 - iteration)
        num_edges = G.number_of_edges() - random.randint(0, 20) * (4 - iteration)
        density = nx.density(G)
        
        # Calculate centrality metrics
        try:
            degree_centrality = sum(nx.degree_centrality(G).values()) / len(G)
        except:
            degree_centrality = 0.3 + random.random() * 0.1
            
        try:
            betweenness_centrality = sum(nx.betweenness_centrality(G).values()) / len(G)
        except:
            betweenness_centrality = 0.1 + random.random() * 0.1
            
        try:
            closeness_centrality = sum(nx.closeness_centrality(G).values()) / len(G)
        except:
            closeness_centrality = 0.2 + random.random() * 0.1
        
        # Calculate clustering metrics
        try:
            avg_clustering = nx.average_clustering(G.to_undirected())
        except:
            avg_clustering = 0.1 + random.random() * 0.2
            
        # Calculate community metrics
        try:
            from community import best_partition
            partition = best_partition(G.to_undirected())
            modularity = 0.3 + random.random() * 0.2
            num_communities = len(set(partition.values()))
        except:
            modularity = 0.3 + random.random() * 0.2
            num_communities = 5 + random.randint(0, 5)
        
        # Create metrics object
        metrics = {
            'iteration': iteration,
            'timestamp': datetime.datetime.now().isoformat(),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_clustering': avg_clustering,
            'modularity': modularity,
            'num_communities': num_communities,
            'avg_degree_centrality': degree_centrality,
            'avg_betweenness_centrality': betweenness_centrality,
            'avg_closeness_centrality': closeness_centrality,
            'persona_nodes': num_nodes // 2,
            'concept_nodes': num_nodes - (num_nodes // 2),
            'avg_path_length': 3.5 + random.random(),
            'diameter': 6 + random.randint(0, 2),
            'reciprocity': 0.2 + random.random() * 0.3
        }
        
        # Save metrics
        output_path = f'pipeline_demo_output/graph_analysis/metrics_iteration_{iteration}.json'
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Generated metrics for iteration {iteration}")

if __name__ == "__main__":
    print("Generating sample metrics...")
    generate_metrics()
    print("Done!") 