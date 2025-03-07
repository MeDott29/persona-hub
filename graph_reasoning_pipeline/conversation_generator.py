import networkx as nx
import json
import random
import logging
import os
import datetime
from typing import List, Dict, Any, Tuple, Optional
import openai
from tqdm import tqdm
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv.load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
CENTRALITY_THRESHOLD = 0.01
KNOWLEDGE_GAP_THRESHOLD = 0.5

def select_similar_persona_pairs(G: nx.Graph, num_pairs: int = 10) -> List[Tuple[str, str]]:
    """
    Select pairs of personas with high similarity for early iterations.
    
    Args:
        G: NetworkX graph with personas as nodes
        num_pairs: Number of pairs to select
        
    Returns:
        List of persona ID pairs
    """
    # Get all edges with their weights
    edges = [(u, v, data['weight']) for u, v, data in G.edges(data=True) 
             if 'weight' in data and G.nodes[u]['type'] == 'persona' and G.nodes[v]['type'] == 'persona']
    
    # Sort by weight (similarity) in descending order
    edges.sort(key=lambda x: x[2], reverse=True)
    
    # Select top pairs
    pairs = [(u, v) for u, v, _ in edges[:num_pairs]]
    
    logger.info(f"Selected {len(pairs)} similar persona pairs")
    return pairs

def select_bridge_persona_pairs(G: nx.Graph, num_pairs: int = 10) -> List[Tuple[str, str]]:
    """
    Select potential bridge personas for later iterations.
    
    Args:
        G: NetworkX graph with personas as nodes
        num_pairs: Number of pairs to select
        
    Returns:
        List of persona ID pairs
    """
    pairs = []
    
    # Only consider persona nodes
    persona_nodes = [node for node, attrs in G.nodes(data=True) 
                    if attrs.get('type') == 'persona']
    
    if len(persona_nodes) < 2:
        logger.warning("Not enough persona nodes to select bridge pairs")
        return pairs
    
    # Compute betweenness centrality
    try:
        centrality = nx.betweenness_centrality(
            G.subgraph(persona_nodes), 
            k=min(100, len(persona_nodes))
        )
    except Exception as e:
        logger.warning(f"Error computing betweenness centrality: {e}")
        # Fallback to degree centrality
        centrality = nx.degree_centrality(G.subgraph(persona_nodes))
    
    # Community detection
    try:
        import community
        communities = community.best_partition(G.subgraph(persona_nodes))
    except Exception as e:
        logger.warning(f"Error in community detection: {e}")
        # Fallback: assign random communities
        communities = {node: random.randint(0, 4) for node in persona_nodes}
    
    # Find potential bridge personas (high betweenness, connections across communities)
    bridge_personas = [p for p, c in centrality.items() 
                      if c > CENTRALITY_THRESHOLD]
    
    # Match bridge personas with personas from different communities
    for bridge in bridge_personas:
        if len(pairs) >= num_pairs:
            break
            
        bridge_community = communities.get(bridge, -1)
        candidates = [p for p in persona_nodes if (
            p != bridge and
            communities.get(p, -2) != bridge_community
        )]
        
        if not candidates:
            continue
            
        # Select a random candidate from a different community
        candidate = random.choice(candidates)
        pairs.append((bridge, candidate))
    
    # If we don't have enough pairs, add some random pairs
    while len(pairs) < num_pairs and len(persona_nodes) >= 2:
        p1, p2 = random.sample(persona_nodes, 2)
        if (p1, p2) not in pairs and (p2, p1) not in pairs:
            pairs.append((p1, p2))
    
    logger.info(f"Selected {len(pairs)} bridge persona pairs")
    return pairs

def knowledge_gap_score(G: nx.Graph, persona1: str, persona2: str) -> float:
    """
    Compute a score representing the potential knowledge gap between two personas.
    
    Args:
        G: NetworkX graph
        persona1: First persona ID
        persona2: Second persona ID
        
    Returns:
        Knowledge gap score (higher means more potential for novel insights)
    """
    # Get node attributes
    p1_attrs = G.nodes[persona1]
    p2_attrs = G.nodes[persona2]
    
    # Different domains = higher score
    domain_diff = 1.0 if p1_attrs.get('domain') != p2_attrs.get('domain') else 0.0
    
    # Different skills = higher score
    p1_skills = set(p1_attrs.get('skills', []))
    p2_skills = set(p2_attrs.get('skills', []))
    skill_overlap = len(p1_skills.intersection(p2_skills)) / max(1, len(p1_skills.union(p2_skills)))
    skill_diff = 1.0 - skill_overlap
    
    # Combine scores (domain difference is weighted more heavily)
    score = (0.7 * domain_diff) + (0.3 * skill_diff)
    
    return score

def get_persona_description(G: nx.Graph, persona_id: str) -> str:
    """
    Get a detailed description of a persona from the graph.
    
    Args:
        G: NetworkX graph
        persona_id: Persona ID
        
    Returns:
        Detailed persona description
    """
    attrs = G.nodes[persona_id]
    
    # Get the original persona description
    original_data = attrs.get('original_data', {})
    if isinstance(original_data, str):
        try:
            original_data = json.loads(original_data)
        except json.JSONDecodeError:
            original_data = {}
    
    description = original_data.get('persona', attrs.get('expertise', ''))
    
    # Add domain information
    domain = attrs.get('domain', 'unknown')
    
    # Format the description
    formatted = f"Persona: {description}\nDomain: {domain}\n"
    
    # Add skills if available
    skills = attrs.get('skills', [])
    if skills:
        if isinstance(skills, str):
            try:
                skills = json.loads(skills)
            except json.JSONDecodeError:
                skills = []
        
        if skills:
            formatted += f"Skills: {', '.join(skills[:10])}\n"
    
    return formatted

def identify_potential_overlaps(G: nx.Graph, persona1_id: str, persona2_id: str) -> List[str]:
    """
    Identify potential overlapping interests between two personas.
    
    Args:
        G: NetworkX graph
        persona1_id: First persona ID
        persona2_id: Second persona ID
        
    Returns:
        List of potential overlapping interests
    """
    # Get node attributes
    p1_attrs = G.nodes[persona1_id]
    p2_attrs = G.nodes[persona2_id]
    
    # Get skills
    p1_skills = set(p1_attrs.get('skills', []))
    p2_skills = set(p2_attrs.get('skills', []))
    
    if isinstance(p1_skills, str):
        try:
            p1_skills = set(json.loads(p1_skills))
        except json.JSONDecodeError:
            p1_skills = set()
    
    if isinstance(p2_skills, str):
        try:
            p2_skills = set(json.loads(p2_skills))
        except json.JSONDecodeError:
            p2_skills = set()
    
    # Find overlapping skills
    overlaps = list(p1_skills.intersection(p2_skills))
    
    # If no direct overlaps, suggest potential areas based on domains
    if not overlaps:
        p1_domain = p1_attrs.get('domain', 'unknown')
        p2_domain = p2_attrs.get('domain', 'unknown')
        
        # Domain-specific potential overlaps
        domain_overlaps = {
            ('technology', 'science'): ['data analysis', 'research methods', 'innovation'],
            ('technology', 'medicine'): ['medical technology', 'health informatics', 'biotech'],
            ('science', 'medicine'): ['medical research', 'biology', 'clinical studies'],
            ('business', 'technology'): ['tech entrepreneurship', 'digital transformation', 'innovation'],
            ('arts', 'technology'): ['digital art', 'creative technology', 'design thinking'],
            ('education', 'technology'): ['educational technology', 'online learning', 'digital literacy'],
            ('politics', 'technology'): ['tech policy', 'digital governance', 'cybersecurity'],
        }
        
        # Check for domain pair overlaps
        domain_pair = (p1_domain, p2_domain)
        reverse_pair = (p2_domain, p1_domain)
        
        if domain_pair in domain_overlaps:
            overlaps = domain_overlaps[domain_pair]
        elif reverse_pair in domain_overlaps:
            overlaps = domain_overlaps[reverse_pair]
        else:
            # Generic overlaps for any domain combination
            overlaps = ['interdisciplinary approaches', 'knowledge sharing', 'innovative thinking']
    
    return overlaps[:3]  # Return top 3 overlaps

def create_conversation_prompt(G: nx.Graph, persona1_id: str, persona2_id: str) -> str:
    """
    Create a prompt for generating a conversation between two personas.
    
    Args:
        G: NetworkX graph
        persona1_id: First persona ID
        persona2_id: Second persona ID
        
    Returns:
        Conversation prompt for the LLM
    """
    # Get detailed persona descriptions
    p1_desc = get_persona_description(G, persona1_id)
    p2_desc = get_persona_description(G, persona2_id)
    
    # Generate potential overlapping interests
    potential_overlaps = identify_potential_overlaps(G, persona1_id, persona2_id)
    
    # Create the prompt template
    prompt = f"""
    A conversation occurs between the following two individuals,  
    please predict the exchange.
    
    Avoid producing a contrived, overly polished exchange. Please remember that humans, especially experts from widely differing fields meeting in an informal conversational setting as in this one, typically engage in a more exploratory and less explicitly analogical manner.
    
    Let's try, keeping the dialogue realistic, relaxed, authentic, and human, lengthy but without an explicit 'goal' or 'point' though at one point they should touch on something typically unknown by the average expert in their fields, and it should color the discussion lightly after.
    
    {p1_desc}
    
    {p2_desc}
    
    {f"These experts might find common ground in: {', '.join(potential_overlaps)}" if potential_overlaps else ""}
    """
    
    return prompt

def query_gpt4o(prompt: str, max_retries: int = 3) -> str:
    """
    Query GPT-4o with the given prompt.
    
    Args:
        prompt: The prompt to send to GPT-4o
        max_retries: Maximum number of retries on failure
        
    Returns:
        The model's response
    """
    client = openai.OpenAI()
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Error querying GPT-4o (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
    
    return "Error generating conversation."

def generate_conversations(G: nx.Graph, iteration: int, num_conversations: int = 10, 
                          output_dir: str = "conversations") -> List[Dict[str, Any]]:
    """
    Generate conversations between selected persona pairs.
    
    Args:
        G: NetworkX graph with personas
        iteration: Current iteration number
        num_conversations: Number of conversations to generate
        output_dir: Directory to save conversations
        
    Returns:
        List of generated conversations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select persona pairs based on current graph structure
    if iteration < 10:
        # Early iterations: Focus on high similarity for establishing core connections
        pairs = select_similar_persona_pairs(G, num_conversations)
    else:
        # Later iterations: Focus on potential bridge connections
        pairs = select_bridge_persona_pairs(G, num_conversations)
    
    conversations = []
    for persona1_id, persona2_id in tqdm(pairs, desc="Generating conversations"):
        # Create conversation prompt
        prompt = create_conversation_prompt(G, persona1_id, persona2_id)
        
        # Generate conversation
        try:
            response = query_gpt4o(prompt)
            
            # Create conversation object
            conversation = {
                'personas': [persona1_id, persona2_id],
                'persona1_desc': get_persona_description(G, persona1_id),
                'persona2_desc': get_persona_description(G, persona2_id),
                'prompt': prompt,
                'content': response,
                'iteration': iteration,
                'timestamp': str(datetime.datetime.now())
            }
            
            conversations.append(conversation)
            
            # Save individual conversation
            conversation_file = os.path.join(
                output_dir, 
                f"conversation_{iteration}_{persona1_id}_{persona2_id}.json"
            )
            with open(conversation_file, 'w') as f:
                json.dump(conversation, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error generating conversation between {persona1_id} and {persona2_id}: {e}")
    
    # Save all conversations for this iteration
    all_conversations_file = os.path.join(output_dir, f"conversations_iteration_{iteration}.json")
    with open(all_conversations_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    logger.info(f"Generated {len(conversations)} conversations for iteration {iteration}")
    return conversations

if __name__ == "__main__":
    from knowledge_graph import load_graph
    
    # Example usage
    G = load_graph("knowledge_graph_initial.gexf")
    conversations = generate_conversations(G, iteration=0, num_conversations=5)
    print(f"Generated {len(conversations)} conversations") 