import networkx as nx
import json
import logging
import os
import datetime
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import community
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def update_knowledge_graph(G: nx.Graph, knowledge_units: List[Dict[str, Any]]) -> nx.Graph:
    """
    Update the knowledge graph with extracted knowledge units.
    
    Args:
        G: NetworkX graph to update
        knowledge_units: List of knowledge unit dictionaries
        
    Returns:
        Updated NetworkX graph
    """
    nodes_added = 0
    edges_added = 0
    insights_added = 0
    
    for unit in tqdm(knowledge_units, desc="Updating graph"):
        # Add concepts as nodes
        for concept in unit.get('concepts', []):
            concept_name = concept.get('name', '')
            if not concept_name:
                continue
                
            # Check if the concept already exists
            if concept_name not in G:
                G.add_node(
                    concept_name,
                    type='concept',
                    description=concept.get('description', ''),
                    domain=concept.get('domain', 'unknown'),
                    creation_timestamp=str(datetime.datetime.now()),
                    source_conversation=unit.get('source_conversation', [])
                )
                nodes_added += 1
            else:
                # Update existing concept with new information
                if 'description' in concept and concept['description']:
                    current_desc = G.nodes[concept_name].get('description', '')
                    if not current_desc or len(concept['description']) > len(current_desc):
                        G.nodes[concept_name]['description'] = concept['description']
                
                # Add source conversation if not already present
                if 'source_conversation' in unit:
                    if 'source_conversation' not in G.nodes[concept_name]:
                        G.nodes[concept_name]['source_conversation'] = []
                    
                    for source in unit['source_conversation']:
                        if source not in G.nodes[concept_name]['source_conversation']:
                            G.nodes[concept_name]['source_conversation'].append(source)
        
        # Add relationships as edges
        for rel in unit.get('relationships', []):
            source = rel.get('source', '')
            target = rel.get('target', '')
            rel_type = rel.get('type', 'related')
            
            if not source or not target:
                continue
                
            # Ensure both nodes exist
            if source not in G:
                G.add_node(
                    source,
                    type='concept',
                    creation_timestamp=str(datetime.datetime.now()),
                    source_conversation=unit.get('source_conversation', [])
                )
                nodes_added += 1
                
            if target not in G:
                G.add_node(
                    target,
                    type='concept',
                    creation_timestamp=str(datetime.datetime.now()),
                    source_conversation=unit.get('source_conversation', [])
                )
                nodes_added += 1
            
            # Add or update the edge
            if G.has_edge(source, target):
                # Update existing edge
                if 'evidence' in rel and rel['evidence']:
                    if 'evidence' not in G.edges[source, target]:
                        G.edges[source, target]['evidence'] = []
                    
                    G.edges[source, target]['evidence'].append(rel['evidence'])
                
                # Update relationship type if not already set
                if 'type' not in G.edges[source, target] and rel_type:
                    G.edges[source, target]['type'] = rel_type
                    
                # Update description if not already set
                if 'description' not in G.edges[source, target] and 'description' in rel:
                    G.edges[source, target]['description'] = rel['description']
            else:
                # Add new edge
                G.add_edge(
                    source,
                    target,
                    type=rel_type,
                    description=rel.get('description', ''),
                    evidence=[rel.get('evidence', '')] if 'evidence' in rel else [],
                    creation_timestamp=str(datetime.datetime.now()),
                    source_conversation=unit.get('source_conversation', [])
                )
                edges_added += 1
        
        # Tag insights to relevant nodes
        for insight in unit.get('insights', []):
            insight_desc = insight.get('description', '')
            if not insight_desc:
                continue
                
            related_concepts = insight.get('related_concepts', [])
            
            # Add insight to related concepts
            for concept in related_concepts:
                if concept in G:
                    if 'insights' not in G.nodes[concept]:
                        G.nodes[concept]['insights'] = []
                    
                    # Add the insight with metadata
                    insight_with_metadata = {
                        'description': insight_desc,
                        'evidence': insight.get('evidence', ''),
                        'novelty_score': insight.get('novelty_score', 0.5),
                        'timestamp': str(datetime.datetime.now()),
                        'source_conversation': unit.get('source_conversation', [])
                    }
                    
                    G.nodes[concept]['insights'].append(insight_with_metadata)
                    insights_added += 1
    
    logger.info(f"Graph update complete: {nodes_added} nodes added, {edges_added} edges added, {insights_added} insights added")
    return G

def analyze_graph_evolution(G: nx.Graph, output_dir: str = "graph_analysis") -> Dict[str, Any]:
    """
    Analyze the evolution of the knowledge graph.
    
    Args:
        G: NetworkX graph to analyze
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary of graph metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute graph metrics
    metrics = {}
    
    # Basic graph statistics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Node type statistics
    node_types = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    metrics['node_types'] = node_types
    
    # Edge type statistics
    edge_types = {}
    for u, v, attrs in G.edges(data=True):
        edge_type = attrs.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    metrics['edge_types'] = edge_types
    
    # Connected components
    components = list(nx.connected_components(G))
    metrics['num_components'] = len(components)
    metrics['largest_component_size'] = len(max(components, key=len))
    
    # Centrality measures (for a subset of nodes if the graph is large)
    if G.number_of_nodes() < 10000:
        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
        metrics['max_degree_centrality'] = max(degree_centrality.values())
        
        # Top nodes by degree centrality
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics['top_nodes_by_centrality'] = [
            {'node': node, 'centrality': centrality} for node, centrality in top_nodes
        ]
        
        # Betweenness centrality (can be slow for large graphs)
        if G.number_of_nodes() < 1000:
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
            metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
            metrics['max_betweenness_centrality'] = max(betweenness_centrality.values())
            
            # Top bridge nodes
            top_bridges = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            metrics['top_bridge_nodes'] = [
                {'node': node, 'betweenness': centrality} for node, centrality in top_bridges
            ]
    
    # Community detection
    try:
        communities = community.best_partition(G)
        metrics['num_communities'] = len(set(communities.values()))
        metrics['modularity'] = community.modularity(communities, G)
        
        # Community sizes
        community_sizes = {}
        for node, comm_id in communities.items():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
        metrics['community_sizes'] = community_sizes
    except Exception as e:
        logger.warning(f"Error in community detection: {e}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f"graph_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metrics_file, 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            elif isinstance(value, np.number):
                metrics_json[key] = value.item()
            else:
                metrics_json[key] = value
        
        json.dump(metrics_json, f, indent=2)
    
    # Visualize the graph (if not too large)
    if G.number_of_nodes() < 500:
        try:
            plt.figure(figsize=(12, 12))
            
            # Use community detection for node colors
            if 'num_communities' in metrics:
                node_colors = [communities.get(node, 0) for node in G.nodes()]
            else:
                # Color by node type
                color_map = {'persona': 'blue', 'concept': 'green'}
                node_colors = [color_map.get(G.nodes[node].get('type', 'unknown'), 'gray') for node in G.nodes()]
            
            # Node size based on degree
            node_size = [10 + 100 * nx.degree(G, node) for node in G.nodes()]
            
            # Draw the graph
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx(
                G, 
                pos=pos, 
                node_color=node_colors, 
                node_size=node_size,
                with_labels=False,
                alpha=0.7,
                edge_color='gray',
                width=0.5
            )
            
            # Save the visualization
            plt.title(f"Knowledge Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"graph_visualization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Error visualizing graph: {e}")
    
    return metrics

def load_knowledge_units(knowledge_dir: str, iteration: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load knowledge units from files.
    
    Args:
        knowledge_dir: Directory containing knowledge unit files
        iteration: Specific iteration to load (None for all)
        
    Returns:
        List of knowledge unit dictionaries
    """
    knowledge_units = []
    
    # Find all knowledge unit files
    for filename in os.listdir(knowledge_dir):
        if not filename.endswith('.json'):
            continue
            
        if iteration is not None and 'iteration' in filename:
            try:
                file_iteration = int(filename.split('_')[1])
                if file_iteration != iteration:
                    continue
            except (IndexError, ValueError):
                pass
            
        file_path = os.path.join(knowledge_dir, filename)
        try:
            with open(file_path, 'r') as f:
                if filename.startswith('knowledge_units_'):
                    # This is a file with multiple knowledge units
                    units = json.load(f)
                    if isinstance(units, list):
                        knowledge_units.extend(units)
                else:
                    # This is a file with a single knowledge unit
                    unit = json.load(f)
                    knowledge_units.append(unit)
        except Exception as e:
            logger.warning(f"Error loading knowledge units from {file_path}: {e}")
    
    logger.info(f"Loaded {len(knowledge_units)} knowledge units")
    return knowledge_units

if __name__ == "__main__":
    import datetime
    from knowledge_graph import load_graph
    
    # Example usage
    G = load_graph("knowledge_graph_initial.gexf")
    knowledge_units = load_knowledge_units("knowledge_units")
    G = update_knowledge_graph(G, knowledge_units)
    metrics = analyze_graph_evolution(G)
    print("Graph metrics:", metrics) 