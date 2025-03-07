#!/bin/bash

# Run the Streamlit app for viewing the graph reasoning pipeline results

# Set default output directory
DEFAULT_OUTPUT_DIR="pipeline_demo_output"

# Ensure the required dependencies are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Print information about the app
echo ""
echo "==================================================================="
echo "  PersonaHub Graph Reasoning Pipeline Viewer"
echo "==================================================================="
echo ""
echo "This app allows you to explore the knowledge graph and data generated"
echo "by the graph reasoning pipeline. You can view:"
echo ""
echo "  - The interactive knowledge graph visualization"
echo "  - Generated conversations between personas"
echo "  - Extracted knowledge units"
echo "  - Graph metrics and analysis"
echo ""
echo "When the app starts, set the 'Pipeline Output Directory' in the sidebar"
echo "to '$DEFAULT_OUTPUT_DIR' or the directory where your pipeline output is stored."
echo ""
echo "Starting the Streamlit app..."
echo "==================================================================="
echo ""

# Run the Streamlit app
streamlit run graph_viewer_app.py 