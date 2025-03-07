# PersonaHub Elite Personas + Agentic Deep Graph Reasoning Pipeline

This project extends the [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) with an Agentic Deep Graph Reasoning Pipeline that leverages elite personas to create self-organizing knowledge networks through iterative conversation generation and knowledge extraction.

## Overview

The Agentic Deep Graph Reasoning Pipeline implements a graph-based reasoning system that:

1. Represents personas, concepts, and their relationships in a knowledge graph
2. Generates conversations between personas to explore knowledge connections
3. Extracts structured knowledge from these conversations
4. Updates the knowledge graph with new insights
5. Analyzes the graph to guide the next iteration of conversations

Through this iterative process, the pipeline enables the emergence of complex knowledge structures without predefined ontologies, allowing for the discovery of novel connections and insights across domains.

## Conceptual Framework

The pipeline leverages the core idea of using iterative graph-based reasoning to create self-organizing knowledge networks where concepts and relationships emerge naturally without predefined ontologies.

### Key Components

1. **Graph-Based Knowledge Representation**
   - Personas as nodes with their expertise as properties
   - Conversations as edges with discovered insights as properties
   - Knowledge domains as higher-level organizing structures

2. **Iterative Reasoning Process**
   ```
   Initialize Graph → Generate Conversations → Extract Insights → 
   Update Graph → Identify New Connections → Select Next Personas → Repeat
   ```

3. **Self-Organization Mechanisms**
   - Allow natural emergence of "hub" concepts and "bridge" personas
   - Track betweenness centrality to identify interdisciplinary connectors
   - Monitor graph modularity to detect knowledge domain formation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/persona-hub.git
   cd persona-hub
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

To run a demo of the graph reasoning pipeline:

```bash
./run_graph_pipeline_demo.sh
```

This will run a small-scale demo of the pipeline with a limited number of iterations and personas.

## Usage

### Running the Full Pipeline

To run the complete pipeline with custom settings:

```bash
python -m graph_reasoning_pipeline.pipeline \
  --personas_file data/persona.jsonl \
  --iterations 10 \
  --initial_personas 1000 \
  --personas_per_iteration 50 \
  --conversations_per_iteration 10 \
  --output_dir pipeline_output \
  --checkpoint_interval 1
```

### Command-line Arguments

- `--personas_file`: Path to the personas JSONL file (default: "data/persona.jsonl")
- `--iterations`: Number of iterations to run (default: 10)
- `--initial_personas`: Number of personas to start with (default: 1000)
- `--personas_per_iteration`: Number of new personas to add per expansion (default: 50)
- `--conversations_per_iteration`: Number of conversations to generate per iteration (default: 10)
- `--output_dir`: Directory to save output (default: "pipeline_output")
- `--checkpoint_interval`: How often to save checkpoints (default: 1)
- `--resume_from`: Path to a checkpoint to resume from (optional)

## Pipeline Architecture

### 1. Knowledge Graph Initialization

The pipeline starts by initializing a knowledge graph with a subset of elite personas from PersonaHub. Each persona is added as a node with attributes such as expertise, skills, and domain. Initial connections between personas are created based on expertise similarity.

```python
# Example: Initialize the knowledge graph
from graph_reasoning_pipeline.knowledge_graph import load_personas, initialize_knowledge_graph

personas = load_personas("data/persona.jsonl", limit=1000)
G = initialize_knowledge_graph(personas)
```

### 2. Conversation Generation

In each iteration, the pipeline selects pairs of personas based on the current graph structure and generates conversations between them using GPT-4o.

```python
# Example: Generate conversations
from graph_reasoning_pipeline.conversation_generator import generate_conversations

conversations = generate_conversations(G, iteration=0, num_conversations=10)
```

### 3. Knowledge Extraction

The pipeline extracts structured knowledge from the generated conversations, including key concepts, relationships, and insights.

```python
# Example: Extract knowledge
from graph_reasoning_pipeline.knowledge_extractor import extract_knowledge

knowledge_units = extract_knowledge(conversations)
```

### 4. Graph Update and Analysis

The knowledge graph is updated with the extracted knowledge and analyzed to guide the next iteration.

```python
# Example: Update and analyze the graph
from graph_reasoning_pipeline.graph_updater import update_knowledge_graph, analyze_graph_evolution

G = update_knowledge_graph(G, knowledge_units)
metrics = analyze_graph_evolution(G)
```

## Output

The pipeline generates the following outputs:

1. **Knowledge Graph**: A NetworkX graph representing personas, concepts, and their relationships
2. **Conversations**: Generated conversations between personas
3. **Knowledge Units**: Structured knowledge extracted from conversations
4. **Graph Metrics**: Analysis of the knowledge graph evolution
5. **Visualizations**: Visual representations of the knowledge graph

## Visualization

The knowledge graph can be visualized using tools like [Gephi](https://gephi.org/). The pipeline saves the graph in GEXF format, which can be directly imported into Gephi for visualization and analysis.

## Extending the Pipeline

The pipeline can be extended in several ways:

1. **Custom Persona Selection**: Implement custom strategies for selecting persona pairs
2. **Alternative Knowledge Extraction**: Use different models or techniques for knowledge extraction
3. **Domain-Specific Analysis**: Add domain-specific metrics and analysis
4. **Integration with Other Systems**: Connect the pipeline to other knowledge systems or databases

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) for providing the persona dataset
- [NetworkX](https://networkx.org/) for the graph data structure and algorithms
- [OpenAI](https://openai.com/) for the GPT-4o model used for conversation generation and knowledge extraction 