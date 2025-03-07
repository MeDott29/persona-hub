import networkx as nx
import json
import itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import community
import os
import jsonlines
from typing import List, Dict, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SIMILARITY_THRESHOLD = 0.3
CENTRALITY_THRESHOLD = 0.01

def load_personas(file_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    Load personas from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing personas
        limit: Maximum number of personas to load (None for all)
        
    Returns:
        List of persona dictionaries
    """
    personas = []
    try:
        with jsonlines.open(file_path) as reader:
            for i, obj in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                if 'persona' in obj:
                    # Add a unique ID if not present
                    if 'id' not in obj:
                        obj['id'] = f"persona_{i}"
                    personas.append(obj)
    except Exception as e:
        logger.error(f"Error loading personas from {file_path}: {e}")
        raise
    
    logger.info(f"Loaded {len(personas)} personas from {file_path}")
    return personas

def extract_expertise_and_skills(persona: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Extract expertise and skills from a persona description.
    
    Args:
        persona: Persona dictionary with 'persona' key
        
    Returns:
        Tuple of (expertise, skills list)
    """
    # This is a simple implementation - in a real system, you might use
    # more sophisticated NLP techniques or LLM-based extraction
    description = persona['persona']
    
    # Extract expertise (assuming it's the main description)
    expertise = description
    
    # Extract skills (this is a placeholder - in reality, you'd use NLP or LLMs)
    # For now, we'll just use some keywords from the description
    words = description.lower().split()
    skills = [word for word in words if len(word) > 5]  # Simple heuristic
    
    return expertise, skills

def classify_domain(persona: Dict[str, Any]) -> str:
    """
    Classify a persona into a knowledge domain.
    
    Args:
        persona: Persona dictionary
        
    Returns:
        Domain classification string
    """
    # This is a placeholder - in a real system, you'd use a more sophisticated
    # classification approach, possibly with a pre-trained classifier
    description = persona['persona'].lower()
    
    domains = {
        'technology': ['software', 'hardware', 'programming', 'developer', 'engineer', 'computer', 'tech'],
        'science': ['scientist', 'physics', 'chemistry', 'biology', 'research', 'laboratory'],
        'medicine': ['doctor', 'medical', 'health', 'healthcare', 'physician', 'surgeon'],
        'arts': ['artist', 'writer', 'musician', 'creative', 'designer', 'author'],
        'business': ['business', 'entrepreneur', 'executive', 'manager', 'finance', 'marketing'],
        'education': ['teacher', 'professor', 'educator', 'academic', 'school', 'university'],
        'politics': ['political', 'policy', 'government', 'diplomat', 'international'],
    }
    
    # Count domain keywords in the description
    domain_scores = {domain: 0 for domain in domains}
    for domain, keywords in domains.items():
        for keyword in keywords:
            if keyword in description:
                domain_scores[domain] += 1
    
    # Return the domain with the highest score, or 'other' if none match
    max_domain = max(domain_scores.items(), key=lambda x: x[1])
    if max_domain[1] > 0:
        return max_domain[0]
    else:
        return 'other'

def compute_expertise_similarity(persona1: Dict[str, Any], persona2: Dict[str, Any]) -> float:
    """
    Compute similarity between two personas based on their expertise.
    
    Args:
        persona1: First persona dictionary
        persona2: Second persona dictionary
        
    Returns:
        Similarity score between 0 and 1
    """
    # Extract text descriptions
    text1 = persona1['persona']
    text2 = persona2['persona']
    
    # Use TF-IDF vectorization and cosine similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        logger.warning(f"Error computing similarity: {e}")
        return 0.0

def initialize_knowledge_graph(personas: List[Dict[str, Any]]) -> nx.Graph:
    """
    Initialize a knowledge graph with personas as nodes.
    
    Args:
        personas: List of persona dictionaries
        
    Returns:
        NetworkX graph with personas as nodes
    """
    G = nx.Graph()
    
    # Add personas as nodes with their expertise as attributes
    for persona in personas:
        expertise, skills = extract_expertise_and_skills(persona)
        domain = classify_domain(persona)
        
        G.add_node(
            persona['id'],
            type='persona',
            expertise=expertise,
            skills=skills,
            domain=domain,
            original_data=persona
        )
    
    logger.info(f"Added {len(personas)} personas as nodes to the graph")
    
    # Create initial connections based on domain overlap
    edges_added = 0
    for p1, p2 in itertools.combinations(personas, 2):
        similarity = compute_expertise_similarity(p1, p2)
        if similarity > SIMILARITY_THRESHOLD:
            G.add_edge(p1['id'], p2['id'], weight=similarity, type='similarity')
            edges_added += 1
    
    logger.info(f"Added {edges_added} initial edges based on expertise similarity")
    
    return G

def analyze_graph(G: nx.Graph) -> Dict[str, Any]:
    """
    Analyze the knowledge graph and compute various metrics.
    
    Args:
        G: NetworkX graph to analyze
        
    Returns:
        Dictionary of graph metrics
    """
    metrics = {}
    
    # Basic graph statistics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
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
        
        # Betweenness centrality (can be slow for large graphs)
        if G.number_of_nodes() < 1000:
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
            metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
            metrics['max_betweenness_centrality'] = max(betweenness_centrality.values())
    
    # Community detection
    try:
        communities = community.best_partition(G)
        metrics['num_communities'] = len(set(communities.values()))
        metrics['modularity'] = community.modularity(communities, G)
    except Exception as e:
        logger.warning(f"Error in community detection: {e}")
    
    return metrics

def save_graph(G: nx.Graph, file_path: str) -> None:
    """
    Save the graph to a file.
    
    Args:
        G: NetworkX graph to save
        file_path: Path to save the graph
    """
    # Convert node attributes to serializable format
    for node, attrs in G.nodes(data=True):
        for key, value in attrs.items():
            if isinstance(value, (list, dict)):
                G.nodes[node][key] = json.dumps(value)
    
    # Save the graph
    nx.write_gexf(G, file_path)
    logger.info(f"Saved graph to {file_path}")

def load_graph(file_path: str) -> nx.Graph:
    """
    Load a graph from a file.
    
    Args:
        file_path: Path to the graph file
        
    Returns:
        Loaded NetworkX graph
    """
    G = nx.read_gexf(file_path)
    
    # Convert serialized attributes back to Python objects
    for node, attrs in G.nodes(data=True):
        for key, value in attrs.items():
            if key in ['skills', 'original_data'] and isinstance(value, str):
                try:
                    G.nodes[node][key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
    
    logger.info(f"Loaded graph from {file_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

if __name__ == "__main__":
    # Example usage
    personas = load_personas("data/persona.jsonl", limit=1000)
    G = initialize_knowledge_graph(personas)
    metrics = analyze_graph(G)
    print("Graph metrics:", metrics)
    save_graph(G, "knowledge_graph_initial.gexf") 