import os
import logging
import argparse
import json
import datetime
from typing import Dict, Any, Optional
import time

# Try relative imports first (when imported as a module)
try:
    from .knowledge_graph import (
        load_personas, 
        initialize_knowledge_graph, 
        analyze_graph, 
        save_graph, 
        load_graph
    )
    from .conversation_generator import generate_conversations
    from .knowledge_extractor import extract_knowledge
    from .graph_updater import update_knowledge_graph, analyze_graph_evolution
# Fall back to absolute imports (when run directly)
except ImportError:
    from knowledge_graph import (
        load_personas, 
        initialize_knowledge_graph, 
        analyze_graph, 
        save_graph, 
        load_graph
    )
    from conversation_generator import generate_conversations
    from knowledge_extractor import extract_knowledge
    from graph_updater import update_knowledge_graph, analyze_graph_evolution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_checkpoint(G, conversations, knowledge_units, metrics, iteration, output_dir="checkpoints"):
    """
    Save a checkpoint of the current pipeline state.
    
    Args:
        G: NetworkX graph
        conversations: List of conversations
        knowledge_units: List of knowledge units
        metrics: Graph metrics
        iteration: Current iteration number
        output_dir: Directory to save checkpoints
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create iteration directory
    iteration_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(iteration_dir, exist_ok=True)
    
    # Save graph
    graph_file = os.path.join(iteration_dir, "knowledge_graph.gexf")
    save_graph(G, graph_file)
    
    # Save conversations
    conversations_file = os.path.join(iteration_dir, "conversations.json")
    with open(conversations_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    # Save knowledge units
    knowledge_file = os.path.join(iteration_dir, "knowledge_units.json")
    with open(knowledge_file, 'w') as f:
        json.dump(knowledge_units, f, indent=2)
    
    # Save metrics
    metrics_file = os.path.join(iteration_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved checkpoint for iteration {iteration} to {iteration_dir}")

def expand_graph_with_new_personas(G, personas_file, num_personas=50):
    """
    Expand the graph with new personas.
    
    Args:
        G: NetworkX graph
        personas_file: Path to the personas file
        num_personas: Number of new personas to add
        
    Returns:
        Updated NetworkX graph
    """
    # Get existing persona IDs
    existing_personas = set()
    for node, attrs in G.nodes(data=True):
        if attrs.get('type') == 'persona':
            existing_personas.add(node)
    
    # Load new personas
    all_personas = load_personas(personas_file)
    
    # Filter out existing personas and select new ones
    new_personas = []
    for persona in all_personas:
        if persona['id'] not in existing_personas:
            new_personas.append(persona)
            if len(new_personas) >= num_personas:
                break
    
    logger.info(f"Adding {len(new_personas)} new personas to the graph")
    
    # Initialize a graph with the new personas
    new_graph = initialize_knowledge_graph(new_personas)
    
    # Merge the new graph with the existing one
    G.add_nodes_from(new_graph.nodes(data=True))
    G.add_edges_from(new_graph.edges(data=True))
    
    logger.info(f"Graph expanded to {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def run_pipeline(
    personas_file: str,
    iterations: int = 100,
    personas_per_iteration: int = 50,
    conversations_per_iteration: int = 10,
    output_dir: str = "pipeline_output",
    initial_personas: int = 1000,
    checkpoint_interval: int = 1,
    resume_from: Optional[str] = None
):
    """
    Run the full graph reasoning pipeline.
    
    Args:
        personas_file: Path to the personas file
        iterations: Number of iterations to run
        personas_per_iteration: Number of new personas to add per expansion
        conversations_per_iteration: Number of conversations to generate per iteration
        output_dir: Directory to save output
        initial_personas: Number of personas to start with
        checkpoint_interval: How often to save checkpoints
        resume_from: Path to a checkpoint to resume from
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    conversations_dir = os.path.join(output_dir, "conversations")
    os.makedirs(conversations_dir, exist_ok=True)
    knowledge_dir = os.path.join(output_dir, "knowledge_units")
    os.makedirs(knowledge_dir, exist_ok=True)
    graph_dir = os.path.join(output_dir, "graph_analysis")
    os.makedirs(graph_dir, exist_ok=True)
    
    # Initialize or load the knowledge graph
    start_iteration = 0
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        G = load_graph(os.path.join(resume_from, "knowledge_graph.gexf"))
        
        # Determine the iteration to start from
        try:
            iteration_str = os.path.basename(resume_from).split("_")[1]
            start_iteration = int(iteration_str) + 1
        except (IndexError, ValueError):
            logger.warning("Could not determine iteration from checkpoint path, starting from 0")
    else:
        logger.info(f"Initializing knowledge graph with {initial_personas} personas")
        personas = load_personas(personas_file, limit=initial_personas)
        G = initialize_knowledge_graph(personas)
    
    # Run the pipeline iterations
    for i in range(start_iteration, iterations):
        logger.info(f"Starting iteration {i+1}/{iterations}")
        iteration_start_time = time.time()
        
        # 1. Generate conversations between selected personas
        logger.info(f"Generating {conversations_per_iteration} conversations")
        conversations = generate_conversations(
            G, 
            iteration=i,
            num_conversations=conversations_per_iteration,
            output_dir=conversations_dir
        )
        
        # 2. Extract knowledge from conversations
        logger.info("Extracting knowledge from conversations")
        knowledge_units = extract_knowledge(
            conversations,
            output_dir=knowledge_dir
        )
        
        # 3. Update the knowledge graph
        logger.info("Updating knowledge graph")
        G = update_knowledge_graph(G, knowledge_units)
        
        # 4. Analyze graph to guide next iteration
        logger.info("Analyzing graph evolution")
        metrics = analyze_graph_evolution(G, output_dir=graph_dir)
        
        # 5. Save checkpoint
        if (i + 1) % checkpoint_interval == 0:
            logger.info(f"Saving checkpoint for iteration {i}")
            save_checkpoint(G, conversations, knowledge_units, metrics, i)
        
        # 6. Potentially expand with more personas
        if (i + 1) % 10 == 0:
            logger.info(f"Expanding graph with {personas_per_iteration} new personas")
            G = expand_graph_with_new_personas(G, personas_file, personas_per_iteration)
        
        iteration_time = time.time() - iteration_start_time
        logger.info(f"Completed iteration {i+1}/{iterations} in {iteration_time:.2f} seconds")
    
    # Final analysis and checkpoint
    logger.info("Performing final graph analysis")
    final_metrics = analyze_graph_evolution(G, output_dir=graph_dir)
    
    logger.info("Saving final checkpoint")
    save_checkpoint(G, [], [], final_metrics, iterations)
    
    # Save final graph
    final_graph_path = os.path.join(output_dir, "final_knowledge_graph.gexf")
    save_graph(G, final_graph_path)
    
    logger.info(f"Pipeline complete. Final graph saved to {final_graph_path}")
    return G

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the graph reasoning pipeline")
    parser.add_argument("--personas_file", type=str, default="data/persona.jsonl", help="Path to the personas file")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--initial_personas", type=int, default=1000, help="Number of personas to start with")
    parser.add_argument("--personas_per_iteration", type=int, default=50, help="Number of new personas to add per expansion")
    parser.add_argument("--conversations_per_iteration", type=int, default=10, help="Number of conversations to generate per iteration")
    parser.add_argument("--output_dir", type=str, default="pipeline_output", help="Directory to save output")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="How often to save checkpoints")
    parser.add_argument("--resume_from", type=str, help="Path to a checkpoint to resume from")
    
    args = parser.parse_args()
    
    run_pipeline(
        personas_file=args.personas_file,
        iterations=args.iterations,
        personas_per_iteration=args.personas_per_iteration,
        conversations_per_iteration=args.conversations_per_iteration,
        output_dir=args.output_dir,
        initial_personas=args.initial_personas,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from
    ) 