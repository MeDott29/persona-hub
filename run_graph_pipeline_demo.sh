#!/bin/bash

# Run a demo of the PersonaHub Elite Personas + Agentic Deep Graph Reasoning Pipeline
# This script runs a small-scale demo of the pipeline with a limited number of iterations and personas

# Ensure the required dependencies are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Create output directory
mkdir -p pipeline_demo_output

# Run the pipeline with a small number of iterations and personas
echo "Running graph reasoning pipeline demo..."
python -m graph_reasoning_pipeline.pipeline \
  --personas_file data/persona.jsonl \
  --iterations 3 \
  --initial_personas 100 \
  --personas_per_iteration 20 \
  --conversations_per_iteration 5 \
  --output_dir pipeline_demo_output \
  --checkpoint_interval 1

echo "Demo complete! Results are saved in the pipeline_demo_output directory."
echo "To view the results, check the following files:"
echo "  - pipeline_demo_output/final_knowledge_graph.gexf (open with Gephi or similar tool)"
echo "  - pipeline_demo_output/graph_analysis/ (contains graph metrics and visualizations)"
echo "  - pipeline_demo_output/conversations/ (contains generated conversations)"
echo "  - pipeline_demo_output/knowledge_units/ (contains extracted knowledge)" 