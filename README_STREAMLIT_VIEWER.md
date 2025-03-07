# PersonaHub Graph Reasoning Pipeline Viewer

This Streamlit application provides an interactive interface for exploring and analyzing the results of the PersonaHub Graph Reasoning Pipeline.

## Features

- **Knowledge Graph Visualization**: Interactive visualization of the knowledge graph with node and edge details
- **Conversation Browser**: View and explore the generated conversations between personas
- **Knowledge Unit Explorer**: Browse and filter extracted knowledge units
- **Metrics Dashboard**: Visualize graph metrics and evolution over iterations

## Getting Started

### Prerequisites

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Running the Viewer

To start the Streamlit app, run:

```bash
./run_graph_viewer.sh
```

Or directly with:

```bash
streamlit run graph_viewer_app.py
```

### Using the Viewer

1. **Select the output directory**: By default, the app looks for results in the `pipeline_demo_output` directory. You can change this in the sidebar.

2. **Navigate between sections**: Use the navigation radio buttons in the sidebar to switch between different views:
   - Knowledge Graph: Visualize and explore the knowledge graph
   - Conversations: Browse generated conversations between personas
   - Knowledge Units: Explore extracted knowledge units with filtering options
   - Metrics: View graph metrics and evolution over time

3. **Interact with visualizations**: 
   - In the Knowledge Graph view, you can zoom, pan, and click on nodes to see details
   - In the Metrics view, hover over data points to see exact values

## Troubleshooting

If you encounter any issues:

1. **No data found**: Make sure you've run the pipeline first using `run_graph_pipeline_demo.sh`
2. **Missing dependencies**: Ensure all required packages are installed with `pip install -r requirements.txt`
3. **Visualization issues**: Try refreshing the page or adjusting the browser window size

## Additional Information

- The graph visualization uses Pyvis, which renders an interactive HTML visualization
- For large graphs, the visualization may take a moment to load and render
- You can download the GEXF file from the Knowledge Graph view to open it in tools like Gephi for more advanced analysis 