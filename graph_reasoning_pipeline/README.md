# PersonaHub Elite Personas + Agentic Deep Graph Reasoning Pipeline

This pipeline implements a graph-based reasoning system that leverages elite personas from PersonaHub to create self-organizing knowledge networks through iterative conversation generation and knowledge extraction.

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

## Usage

### Running the Full Pipeline

To run the complete pipeline with default settings:

```bash
python -m graph_reasoning_pipeline.pipeline --personas_file data/persona.jsonl --iterations 10
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

### Running Individual Components

You can also run individual components of the pipeline:

1. Initialize the knowledge graph:
   ```bash
   python -m graph_reasoning_pipeline.knowledge_graph
   ```

2. Generate conversations:
   ```bash
   python -m graph_reasoning_pipeline.conversation_generator
   ```

3. Extract knowledge:
   ```bash
   python -m graph_reasoning_pipeline.knowledge_extractor
   ```

4. Update and analyze the graph:
   ```bash
   python -m graph_reasoning_pipeline.graph_updater
   ```

## Pipeline Architecture

### 1. Knowledge Graph Initialization

The pipeline starts by initializing a knowledge graph with a subset of elite personas from PersonaHub. Each persona is added as a node with attributes such as expertise, skills, and domain. Initial connections between personas are created based on expertise similarity.

### 2. Conversation Generation

In each iteration, the pipeline selects pairs of personas based on the current graph structure. In early iterations, it focuses on similar personas to establish core connections. In later iterations, it selects potential bridge personas to connect different knowledge domains.

### 3. Knowledge Extraction

The pipeline extracts structured knowledge from the generated conversations, including:
- Key concepts discussed
- Relationships between concepts
- Novel insights or facts generated

### 4. Graph Update and Analysis

The knowledge graph is updated with the extracted knowledge, adding new concepts as nodes and relationships as edges. The pipeline then analyzes the updated graph to guide the next iteration, tracking metrics such as:
- Hub formation (node degree distribution)
- Modularity (emergence of clusters/communities)
- Bridge nodes (personas connecting disparate knowledge domains)
- Knowledge path growth (formation of longest shortest paths)

### 5. Persona Selection Strategy

The pipeline uses different strategies to select persona pairs for conversation generation:
- In early iterations, it selects pairs with high similarity to establish core connections
- In later iterations, it focuses on potential bridge personas to connect different knowledge domains

## Output

The pipeline generates the following outputs:

1. **Knowledge Graph**: A NetworkX graph representing personas, concepts, and their relationships
2. **Conversations**: Generated conversations between personas
3. **Knowledge Units**: Structured knowledge extracted from conversations
4. **Graph Metrics**: Analysis of the knowledge graph evolution
5. **Visualizations**: Visual representations of the knowledge graph

## Monitoring and Analysis

The pipeline includes tools to monitor and analyze the self-organizing properties of the knowledge graph:

1. **Track hub formation**: Monitor the evolution of node degree distribution
2. **Measure modularity**: Track the emergence of clusters/communities
3. **Analyze bridge nodes**: Identify personas that connect disparate knowledge domains
4. **Chart knowledge path growth**: Monitor the formation of longest shortest paths

## License

This project is licensed under the MIT License - see the LICENSE file for details. 