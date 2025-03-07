# PersonaHub Graph Reasoning Pipeline

This project implements a graph-based reasoning pipeline that leverages elite personas from PersonaHub to create self-organizing knowledge networks through iterative conversation generation and knowledge extraction.

## Getting Started

### Prerequisites

1. Python 3.8 or higher
2. OpenAI API key (for generating conversations)

### Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file to add your OpenAI API key.

## Running the Pipeline

### Demo Run

To run a small demo of the pipeline without an OpenAI API key:

```bash
./run_graph_pipeline_demo.sh
```

This will run a limited version of the pipeline without generating conversations.

### Full Run

To run the full pipeline with your OpenAI API key:

```bash
./run_graph_pipeline_with_api.sh
```

This will:
1. Load personas from the data file
2. Initialize a knowledge graph
3. Generate conversations between personas
4. Extract knowledge from conversations
5. Update the knowledge graph with new insights
6. Analyze the graph evolution
7. Save the results to the `pipeline_output` directory

## Viewing the Results

To view the results of the pipeline, run the Streamlit app:

```bash
./run_graph_viewer.sh
```

This will start a web interface where you can:
- Visualize the knowledge graph
- Browse generated conversations
- Explore extracted knowledge units
- View graph metrics and evolution

By default, the app looks for results in the `pipeline_demo_output` directory. You can change this in the sidebar to view results from other runs.

## Project Structure

- `graph_reasoning_pipeline/`: Core pipeline implementation
  - `pipeline.py`: Main pipeline orchestration
  - `knowledge_graph.py`: Graph initialization and management
  - `conversation_generator.py`: Conversation generation between personas
  - `knowledge_extractor.py`: Knowledge extraction from conversations
  - `graph_updater.py`: Graph update with new knowledge
- `data/`: Contains persona data
- `graph_viewer_app.py`: Streamlit app for visualizing results
- `run_graph_pipeline_demo.sh`: Script to run a demo of the pipeline
- `run_graph_pipeline_with_api.sh`: Script to run the full pipeline
- `run_graph_viewer.sh`: Script to run the Streamlit viewer

## Troubleshooting

### No Conversations Generated

If no conversations are generated, check:
1. Your OpenAI API key is correctly set in the `.env` file
2. You have sufficient credits in your OpenAI account
3. The API is not experiencing downtime

### Graph Visualization Issues

If the graph visualization doesn't load:
1. Try refreshing the page
2. Check that the graph file exists in the output directory
3. Ensure the graph has nodes and edges

### Missing Data

If data is missing in the Streamlit app:
1. Check that the pipeline completed successfully
2. Verify the correct output directory is selected in the sidebar
3. Look for error messages in the terminal output 