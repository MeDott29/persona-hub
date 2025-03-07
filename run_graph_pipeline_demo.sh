#!/bin/bash

# Run a demo of the PersonaHub Elite Personas + Agentic Deep Graph Reasoning Pipeline
# This script runs a small-scale demo of the pipeline with a limited number of iterations and personas

# Set output directory
OUTPUT_DIR="pipeline_demo_output"

# Ensure the required dependencies are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the pipeline with a small number of iterations and personas
echo "Running graph reasoning pipeline demo..."
python -m graph_reasoning_pipeline.pipeline \
  --personas_file data/persona.jsonl \
  --iterations 3 \
  --initial_personas 100 \
  --personas_per_iteration 20 \
  --conversations_per_iteration 5 \
  --output_dir $OUTPUT_DIR \
  --checkpoint_interval 1

echo "Demo complete! Results are saved in the $OUTPUT_DIR directory."
echo "To view the results, check the following files:"
echo "  - $OUTPUT_DIR/final_knowledge_graph.gexf (open with Gephi or similar tool)"
echo "  - $OUTPUT_DIR/graph_analysis/ (contains graph metrics and visualizations)"
echo "  - $OUTPUT_DIR/conversations/ (contains generated conversations)"
echo "  - $OUTPUT_DIR/knowledge_units/ (contains extracted knowledge)"

# Launch the Streamlit app automatically
echo ""
echo "Launching the Streamlit graph viewer app..."
echo "This will allow you to explore the generated knowledge graph and data interactively."
echo ""

# Run the Streamlit app
streamlit run graph_viewer_app.py 