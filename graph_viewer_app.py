import streamlit as st
import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyvis.network import Network
import tempfile
import base64
from pathlib import Path
import glob

st.set_page_config(
    page_title="PersonaHub Graph Reasoning Pipeline Viewer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_graph(graph_path):
    """Load a NetworkX graph from a GEXF file."""
    try:
        return nx.read_gexf(graph_path)
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return None

def load_conversations(conversations_dir):
    """Load conversations from JSON files in the specified directory."""
    conversations = []
    if os.path.exists(conversations_dir):
        # First try to load individual conversation files
        individual_files = glob.glob(os.path.join(conversations_dir, "conversation_*.json"))
        for file_path in individual_files:
            try:
                with open(file_path, 'r') as f:
                    conversation = json.load(f)
                    conversations.append(conversation)
            except Exception as e:
                st.warning(f"Error loading conversation {file_path}: {e}")
        
        # If no individual files found, try iteration files
        if not conversations:
            iteration_files = glob.glob(os.path.join(conversations_dir, "conversations_iteration_*.json"))
            for file_path in iteration_files:
                try:
                    with open(file_path, 'r') as f:
                        iteration_conversations = json.load(f)
                        if isinstance(iteration_conversations, list):
                            conversations.extend(iteration_conversations)
                except Exception as e:
                    st.warning(f"Error loading conversations from {file_path}: {e}")
    
    return conversations

def load_knowledge_units(knowledge_dir):
    """Load knowledge units from JSON files in the specified directory."""
    knowledge_units = []
    if os.path.exists(knowledge_dir):
        # First try to load individual knowledge unit files
        individual_files = glob.glob(os.path.join(knowledge_dir, "knowledge_*.json"))
        for file_path in individual_files:
            try:
                with open(file_path, 'r') as f:
                    knowledge = json.load(f)
                    knowledge_units.append(knowledge)
            except Exception as e:
                st.warning(f"Error loading knowledge unit {file_path}: {e}")
        
        # If no individual files found, try iteration files
        if not knowledge_units:
            iteration_files = glob.glob(os.path.join(knowledge_dir, "knowledge_units_iteration_*.json"))
            for file_path in iteration_files:
                try:
                    with open(file_path, 'r') as f:
                        iteration_knowledge = json.load(f)
                        if isinstance(iteration_knowledge, list):
                            knowledge_units.extend(iteration_knowledge)
                except Exception as e:
                    st.warning(f"Error loading knowledge units from {file_path}: {e}")
    
    return knowledge_units

def load_graph_metrics(metrics_dir):
    """Load graph metrics from JSON files in the specified directory."""
    metrics = []
    if os.path.exists(metrics_dir):
        for file_path in glob.glob(os.path.join(metrics_dir, "*.json")):
            try:
                with open(file_path, 'r') as f:
                    metric = json.load(f)
                    metrics.append(metric)
            except Exception as e:
                st.warning(f"Error loading metric {file_path}: {e}")
    return metrics

def get_graph_stats(G):
    """Get basic statistics about the graph."""
    if G is None:
        return {}
    
    return {
        "Number of nodes": G.number_of_nodes(),
        "Number of edges": G.number_of_edges(),
        "Average degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "Number of connected components": nx.number_connected_components(G.to_undirected()),
        "Density": nx.density(G),
    }

def visualize_graph_pyvis(G, height=600):
    """Create an interactive visualization of the graph using pyvis."""
    if G is None or G.number_of_nodes() == 0:
        st.warning("No graph data to visualize.")
        return
    
    # Create a pyvis network
    net = Network(height=height, width="100%", notebook=True, directed=True)
    
    # Add nodes with properties
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        
        # Set node color based on type
        if node_type == 'persona':
            color = '#4285F4'  # Blue for personas
        elif node_type == 'concept':
            color = '#EA4335'  # Red for concepts
        else:
            color = '#FBBC05'  # Yellow for other types
        
        # Set node size based on degree
        size = 10 + G.degree(node_id) * 2
        
        # Add node with properties
        net.add_node(
            node_id, 
            label=node_data.get('label', str(node_id)),
            title=f"Type: {node_type}<br>Degree: {G.degree(node_id)}",
            color=color,
            size=size
        )
    
    # Add edges with properties
    for source, target, edge_data in G.edges(data=True):
        weight = edge_data.get('weight', 1.0)
        relation = edge_data.get('relation', '')
        
        # Add edge with properties
        net.add_edge(
            source, 
            target, 
            title=relation,
            width=1 + weight * 2,
            arrowStrikethrough=True
        )
    
    # Set physics options for better visualization
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      }
    }
    """)
    
    # Save and display the graph
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html = f.read()
    
    # Display the graph in an iframe
    st.components.v1.html(html, height=height, scrolling=True)

def plot_metrics_over_time(metrics):
    """Plot graph metrics over iterations."""
    if not metrics:
        st.warning("No metrics data available.")
        return
    
    # Sort metrics by iteration
    metrics = sorted(metrics, key=lambda x: x.get('iteration', 0))
    
    # Extract data for plotting
    iterations = [m.get('iteration', i) for i, m in enumerate(metrics)]
    nodes = [m.get('num_nodes', 0) for m in metrics]
    edges = [m.get('num_edges', 0) for m in metrics]
    density = [m.get('density', 0) for m in metrics]
    avg_clustering = [m.get('avg_clustering', 0) for m in metrics]
    
    # Create plots
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=nodes, mode='lines+markers', name='Nodes'))
    fig.add_trace(go.Scatter(x=iterations, y=edges, mode='lines+markers', name='Edges'))
    
    fig.update_layout(
        title='Graph Growth Over Iterations',
        xaxis_title='Iteration',
        yaxis_title='Count',
        legend_title='Metric',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create second plot for other metrics
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=iterations, y=density, mode='lines+markers', name='Density'))
    fig2.add_trace(go.Scatter(x=iterations, y=avg_clustering, mode='lines+markers', name='Avg. Clustering'))
    
    fig2.update_layout(
        title='Graph Properties Over Iterations',
        xaxis_title='Iteration',
        yaxis_title='Value',
        legend_title='Metric',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

def display_conversation(conversation):
    """Display a conversation in a structured format."""
    if not conversation:
        st.warning("No conversation data to display.")
        return
    
    # Extract conversation metadata
    persona1 = conversation.get('persona1', 'Unknown')
    persona2 = conversation.get('persona2', 'Unknown')
    topic = conversation.get('topic', 'Unknown')
    iteration = conversation.get('iteration', 'Unknown')
    
    # Display conversation header
    st.subheader(f"Conversation between {persona1} and {persona2}")
    st.write(f"**Topic:** {topic}")
    st.write(f"**Iteration:** {iteration}")
    
    # Display messages
    messages = conversation.get('messages', [])
    if not messages:
        # Try alternative format
        content = conversation.get('content', '')
        if content:
            st.text_area("Conversation", content, height=400)
            return
    
    # Display structured messages
    for msg in messages:
        if isinstance(msg, dict):
            speaker = msg.get('speaker', 'Unknown')
            content = msg.get('content', '')
            
            col1, col2 = st.columns([1, 5])
            with col1:
                st.write(f"**{speaker}:**")
            with col2:
                st.write(content)
            st.write("---")
        elif isinstance(msg, str):
            st.write(msg)
            st.write("---")

def display_knowledge_unit(ku):
    """Display a knowledge unit in a structured format."""
    if not ku:
        st.warning("No knowledge unit data to display.")
        return
    
    # Extract knowledge unit metadata
    ku_type = ku.get('type', 'Knowledge')
    source = ku.get('source', 'Unknown')
    confidence = ku.get('confidence', 0)
    iteration = ku.get('iteration', 'Unknown')
    
    # Create expander header
    with st.expander(f"{ku_type} from {source} (Confidence: {confidence:.2f})"):
        # Display relation if available
        if 'subject' in ku and 'predicate' in ku and 'object' in ku:
            st.write(f"**Relation:** {ku['subject']} {ku['predicate']} {ku['object']}")
        
        # Display content if available
        if 'content' in ku:
            st.write(f"**Content:** {ku['content']}")
        
        # Display additional metadata
        st.write(f"**Iteration:** {iteration}")
        st.write(f"**Extracted from:** {ku.get('conversation_id', 'Unknown')}")

def main():
    st.title("PersonaHub Graph Reasoning Pipeline Viewer")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Knowledge Graph", "Conversations", "Knowledge Units", "Metrics"]
    )
    
    # Input for output directory
    output_dir = st.sidebar.text_input(
        "Pipeline Output Directory",
        value="pipeline_demo_output"
    )
    
    if not os.path.exists(output_dir):
        st.sidebar.error(f"Directory {output_dir} does not exist.")
        return
    
    # Load data based on selected page
    if page == "Knowledge Graph":
        st.header("Knowledge Graph Visualization")
        
        # Load the final knowledge graph
        graph_path = os.path.join(output_dir, "final_knowledge_graph.gexf")
        if os.path.exists(graph_path):
            G = load_graph(graph_path)
            
            # Display graph statistics
            st.subheader("Graph Statistics")
            stats = get_graph_stats(G)
            stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
            st.table(stats_df)
            
            # Visualize the graph
            st.subheader("Interactive Graph Visualization")
            visualize_graph_pyvis(G)
            
            # Option to download the graph file
            with open(graph_path, "rb") as file:
                btn = st.download_button(
                    label="Download GEXF File",
                    data=file,
                    file_name="knowledge_graph.gexf",
                    mime="application/octet-stream"
                )
        else:
            st.error(f"Graph file not found at {graph_path}")
    
    elif page == "Conversations":
        st.header("Generated Conversations")
        
        # Load conversations
        conversations_dir = os.path.join(output_dir, "conversations")
        conversations = load_conversations(conversations_dir)
        
        if conversations:
            # Create a selectbox to choose a conversation
            conversation_ids = [f"Conversation {i+1}" for i in range(len(conversations))]
            selected_conversation = st.selectbox("Select a conversation", conversation_ids)
            
            # Display the selected conversation
            idx = conversation_ids.index(selected_conversation)
            conversation = conversations[idx]
            
            # Display the conversation
            display_conversation(conversation)
            
            # Show raw data option
            with st.expander("View raw conversation data"):
                st.json(conversation)
        else:
            st.warning("No conversations found in the specified directory.")
            st.info("If you've run the pipeline, check that the conversations were generated correctly.")
    
    elif page == "Knowledge Units":
        st.header("Extracted Knowledge Units")
        
        # Load knowledge units
        knowledge_dir = os.path.join(output_dir, "knowledge_units")
        knowledge_units = load_knowledge_units(knowledge_dir)
        
        if knowledge_units:
            # Create a dataframe for filtering
            knowledge_df = pd.DataFrame([
                {
                    "ID": i,
                    "Source": ku.get('source', 'Unknown') if isinstance(ku, dict) else 'Unknown',
                    "Type": ku.get('type', 'Unknown') if isinstance(ku, dict) else 'Unknown',
                    "Confidence": float(ku.get('confidence', 0)) if isinstance(ku, dict) else 0.0,
                    "Iteration": int(ku.get('iteration', 0)) if isinstance(ku, dict) else 0
                }
                for i, ku in enumerate(knowledge_units)
            ])
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                selected_types = st.multiselect(
                    "Filter by knowledge type",
                    options=sorted(knowledge_df['Type'].unique()),
                    default=sorted(knowledge_df['Type'].unique())
                )
            
            with col2:
                min_confidence = st.slider(
                    "Minimum confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
            
            # Apply filters
            filtered_df = knowledge_df[
                (knowledge_df['Type'].isin(selected_types)) &
                (knowledge_df['Confidence'] >= min_confidence)
            ]
            
            # Display filtered knowledge units
            if not filtered_df.empty:
                st.write(f"Showing {len(filtered_df)} of {len(knowledge_units)} knowledge units")
                
                for _, row in filtered_df.iterrows():
                    ku = knowledge_units[row['ID']]
                    
                    if isinstance(ku, dict):
                        display_knowledge_unit(ku)
                    else:
                        st.write("Knowledge unit data format is not as expected. Raw data:")
                        st.json(ku)
            else:
                st.warning("No knowledge units match the selected filters.")
        else:
            st.warning("No knowledge units found in the specified directory.")
            st.info("If you've run the pipeline, check that the knowledge extraction was completed correctly.")
    
    elif page == "Metrics":
        st.header("Graph Analysis Metrics")
        
        # Load metrics
        metrics_dir = os.path.join(output_dir, "graph_analysis")
        metrics = load_graph_metrics(metrics_dir)
        
        if metrics:
            # Plot metrics over time
            st.subheader("Graph Evolution")
            plot_metrics_over_time(metrics)
            
            # Display detailed metrics for the latest iteration
            latest_metric = max(metrics, key=lambda x: x.get('iteration', 0))
            
            st.subheader(f"Detailed Metrics (Iteration {latest_metric.get('iteration', 'Unknown')})")
            
            # Create a dataframe for the metrics
            metrics_items = {k: v for k, v in latest_metric.items() if k != 'iteration' and k != 'timestamp'}
            
            # Convert all values to appropriate types
            formatted_items = []
            for k, v in metrics_items.items():
                if isinstance(v, (int, float)):
                    formatted_value = v
                else:
                    # Try to convert to float if it looks like a number
                    try:
                        formatted_value = float(v)
                    except (ValueError, TypeError):
                        formatted_value = v
                
                formatted_items.append((k, formatted_value))
            
            metrics_df = pd.DataFrame(formatted_items, columns=["Metric", "Value"])
            
            # Format numeric values for display
            metrics_df['Display_Value'] = metrics_df['Value'].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
            )
            
            # Display the table with the formatted values
            st.table(metrics_df[["Metric", "Display_Value"]].rename(columns={"Display_Value": "Value"}))
        else:
            st.warning("No metrics data found in the specified directory.")
            st.info("If you've run the pipeline, check that the graph analysis was completed correctly.")

if __name__ == "__main__":
    main() 