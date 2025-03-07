#!/bin/bash

# Run the PersonaHub Elite Personas + Agentic Deep Graph Reasoning Pipeline
# This script runs the pipeline with the OpenAI API key from the .env file

# Check if .env file exists
if [ ! -f .env ]; then
  echo "Error: .env file not found. Please create a .env file with your OpenAI API key."
  echo "You can copy the .env.example file and add your API key:"
  echo "cp .env.example .env"
  echo "Then edit the .env file to add your OpenAI API key."
  exit 1
fi

# Ensure the required dependencies are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Create output directories
echo "Creating output directories..."
mkdir -p pipeline_output
mkdir -p pipeline_output/conversations
mkdir -p pipeline_output/knowledge_units
mkdir -p pipeline_output/graph_analysis
mkdir -p pipeline_output/checkpoints

# Run the pipeline
echo "Running graph reasoning pipeline..."
python -m graph_reasoning_pipeline.pipeline \
  --personas_file data/persona.jsonl \
  --iterations 5 \
  --initial_personas 100 \
  --personas_per_iteration 20 \
  --conversations_per_iteration 5 \
  --output_dir pipeline_output \
  --checkpoint_interval 1

echo "Pipeline complete! Results are saved in the pipeline_output directory."
echo "To view the results, run the Streamlit app with:"
echo "./run_graph_viewer.sh"
echo ""
echo "Then set the Pipeline Output Directory to 'pipeline_output' in the sidebar." 