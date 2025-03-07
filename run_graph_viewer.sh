#!/bin/bash

# Run the Streamlit app for viewing the graph reasoning pipeline results

# Ensure the required dependencies are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting the Streamlit app..."
streamlit run graph_viewer_app.py 