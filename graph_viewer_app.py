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
        # First try to load all_knowledge_units.json which contains all units
        all_units_path = os.path.join(knowledge_dir, "all_knowledge_units.json")
        if os.path.exists(all_units_path):
            try:
                with open(all_units_path, 'r') as f:
                    all_units = json.load(f)
                    if isinstance(all_units, list):
                        knowledge_units.extend(all_units)
                        return knowledge_units
            except Exception as e:
                st.warning(f"Error loading all knowledge units: {e}")
        
        # If all_knowledge_units.json not found or empty, try individual files
        individual_files = glob.glob(os.path.join(knowledge_dir, "knowledge_*.json"))
        for file_path in individual_files:
            try:
                with open(file_path, 'r') as f:
                    knowledge = json.load(f)
                    # Handle both single dict and list of dicts
                    if isinstance(knowledge, list):
                        knowledge_units.extend(knowledge)
                    else:
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

def visualize_graph_pyvis(G, height=1000):
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
        size = 20 + G.degree(node_id) * 3
        
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
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "hover": true,
        "multiselect": true,
        "tooltipDelay": 100
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        }
      }
    }
    """)
    
    # Save and display the graph
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html = f.read()
    
    # Display the graph in an iframe with full width
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
    
    # Create growth plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=iterations, y=nodes, mode='lines+markers', name='Nodes', line=dict(color='#4285F4', width=3)))
    fig1.add_trace(go.Scatter(x=iterations, y=edges, mode='lines+markers', name='Edges', line=dict(color='#EA4335', width=3)))
    
    fig1.update_layout(
        title='Graph Growth Over Iterations',
        xaxis_title='Iteration',
        yaxis_title='Count',
        legend_title='Metric',
        hovermode='x unified',
        height=400,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Create second plot for other metrics
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=iterations, y=density, mode='lines+markers', name='Density', line=dict(color='#FBBC05', width=3)))
    fig2.add_trace(go.Scatter(x=iterations, y=avg_clustering, mode='lines+markers', name='Avg. Clustering', line=dict(color='#34A853', width=3)))
    
    fig2.update_layout(
        title='Graph Properties Over Iterations',
        xaxis_title='Iteration',
        yaxis_title='Value',
        legend_title='Metric',
        hovermode='x unified',
        height=400,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Extract additional metrics if available
    if all('modularity' in m for m in metrics):
        modularity = [m.get('modularity', 0) for m in metrics]
        communities = [m.get('num_communities', 0) for m in metrics]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=iterations, y=modularity, mode='lines+markers', name='Modularity', line=dict(color='#673AB7', width=3)))
        fig3.add_trace(go.Scatter(x=iterations, y=communities, mode='lines+markers', name='Communities', line=dict(color='#FF6D00', width=3), yaxis="y2"))
        
        fig3.update_layout(
            title='Community Structure Over Iterations',
            xaxis_title='Iteration',
            yaxis_title='Modularity',
            yaxis2=dict(
                title='Number of Communities',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig3, use_container_width=True)

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
    
    # Display the conversation messages
    messages = conversation.get('messages', [])
    if messages:
        st.markdown("---")
        for i, message in enumerate(messages):
            speaker = message.get('speaker', 'Unknown')
            content = message.get('content', '')
            
            # Determine which persona is speaking
            if speaker == persona1:
                avatar = "👤"
                align = "left"
                bg_color = "#E6F7FF"  # Light blue
            elif speaker == persona2:
                avatar = "👥"
                align = "right"
                bg_color = "#F0F0F0"  # Light gray
            else:
                avatar = "🤖"
                align = "center"
                bg_color = "#FFFDE7"  # Light yellow
            
            # Create a styled message box
            st.markdown(f"""
            <div style="display: flex; justify-content: {align}; margin-bottom: 10px;">
                <div style="background-color: {bg_color}; padding: 10px; border-radius: 10px; max-width: 80%;">
                    <div style="font-weight: bold;">{avatar} {speaker}</div>
                    <div style="white-space: pre-wrap;">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No messages found in this conversation.")

def display_knowledge_unit(ku):
    """Display a knowledge unit in a structured format."""
    if not ku:
        st.warning("No knowledge unit data to display.")
        return
    
    # Extract knowledge unit metadata
    ku_type = ku.get('type', 'Unknown')
    confidence = ku.get('confidence', 0.0)
    source = ku.get('source', 'Unknown')
    iteration = ku.get('iteration', 'Unknown')
    
    # Create a card-like container for the knowledge unit
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div><strong>Type:</strong> {ku_type}</div>
            <div><strong>Confidence:</strong> <span style="color: {'green' if float(confidence) > 0.7 else 'orange' if float(confidence) > 0.4 else 'red'}">{confidence}</span></div>
        </div>
        <div style="margin-bottom: 10px;"><strong>Content:</strong> {ku.get('content', 'No content available')}</div>
        <div style="color: #666; font-size: 0.9em;">
            <div><strong>Source:</strong> {source}</div>
            <div><strong>Iteration:</strong> {iteration}</div>
            <div><strong>ID:</strong> {ku.get('id', 'Unknown')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            
            # Create columns for layout
            col1, col2 = st.columns([1, 4])  # Increase the ratio to give more space to the graph
            
            with col1:
                # Display graph statistics
                st.subheader("Graph Statistics")
                stats = get_graph_stats(G)
                stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
                st.table(stats_df)
                
                # Add graph visualization controls
                st.subheader("Visualization Controls")
                graph_height = st.slider("Graph Height", min_value=800, max_value=1500, value=1000, step=100)
                
                # Node filtering options
                st.subheader("Filter Nodes")
                node_types = list(set(nx.get_node_attributes(G, 'type').values()))
                selected_node_types = st.multiselect(
                    "Show node types",
                    options=node_types,
                    default=node_types
                )
                
                # Option to download the graph file
                st.subheader("Export")
                with open(graph_path, "rb") as file:
                    btn = st.download_button(
                        label="Download GEXF File",
                        data=file,
                        file_name="knowledge_graph.gexf",
                        mime="application/octet-stream"
                    )
                st.info("You can open the GEXF file with tools like Gephi for advanced visualization and analysis.")
            
            with col2:
                # Visualize the graph
                st.subheader("Interactive Graph Visualization")
                
                # Filter graph based on selected node types if needed
                if len(selected_node_types) < len(node_types):
                    filtered_G = G.copy()
                    nodes_to_remove = [n for n, attrs in G.nodes(data=True) 
                                      if attrs.get('type') not in selected_node_types]
                    filtered_G.remove_nodes_from(nodes_to_remove)
                    visualize_graph_pyvis(filtered_G, height=graph_height)
                    st.caption(f"Showing filtered graph with {filtered_G.number_of_nodes()} nodes and {filtered_G.number_of_edges()} edges")
                else:
                    visualize_graph_pyvis(G, height=graph_height)
                    st.caption(f"Showing complete graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                
                # Add legend
                st.markdown("""
                **Legend:**
                - 🔵 Blue nodes: Personas
                - 🔴 Red nodes: Concepts
                - 🟡 Yellow nodes: Other types
                - Node size indicates importance (degree centrality)
                - Edge thickness indicates relationship strength
                """)
                
                # Add usage tips
                with st.expander("Graph Interaction Tips"):
                    st.markdown("""
                    - **Zoom**: Use mouse wheel or pinch gesture
                    - **Pan**: Click and drag on empty space
                    - **Select nodes**: Click on a node to highlight its connections
                    - **Multi-select**: Hold Ctrl/Cmd while clicking nodes
                    - **View details**: Hover over nodes or edges to see details
                    - **Move nodes**: Click and drag nodes to reposition them
                    - **Stabilize**: Double-click on empty space to stabilize the layout
                    """)
        else:
            st.error(f"Graph file not found at {graph_path}")
            st.info("Please run the pipeline first using the run_graph_pipeline_demo.sh script to generate the graph data.")
    
    elif page == "Conversations":
        st.header("Generated Conversations")
        
        # Load conversations
        conversations_dir = os.path.join(output_dir, "conversations")
        conversations = load_conversations(conversations_dir)
        
        if conversations:
            # Add filters for conversations
            col1, col2 = st.columns(2)
            
            with col1:
                # Get unique iterations
                iterations = sorted(list(set([c.get('iteration', 0) for c in conversations if isinstance(c, dict)])))
                selected_iterations = st.multiselect(
                    "Filter by iteration",
                    options=iterations,
                    default=iterations
                )
            
            with col2:
                # Get unique topics
                topics = sorted(list(set([c.get('topic', 'Unknown') for c in conversations if isinstance(c, dict)])))
                selected_topics = st.multiselect(
                    "Filter by topic",
                    options=topics,
                    default=[]
                )
            
            # Apply filters
            filtered_conversations = conversations
            if selected_iterations:
                filtered_conversations = [c for c in filtered_conversations 
                                         if c.get('iteration') in selected_iterations]
            if selected_topics:
                filtered_conversations = [c for c in filtered_conversations 
                                         if c.get('topic') in selected_topics]
            
            if filtered_conversations:
                # Create a more informative selectbox
                conversation_options = [
                    f"Conversation {i+1}: {c.get('persona1', 'Unknown')} & {c.get('persona2', 'Unknown')} - {c.get('topic', 'Unknown')}"
                    for i, c in enumerate(filtered_conversations)
                ]
                
                selected_conversation = st.selectbox(
                    f"Select a conversation ({len(filtered_conversations)} available)",
                    conversation_options
                )
                
                # Display the selected conversation
                idx = conversation_options.index(selected_conversation)
                conversation = filtered_conversations[idx]
                
                # Display the conversation
                display_conversation(conversation)
                
                # Show raw data option
                with st.expander("View raw conversation data"):
                    st.json(conversation)
            else:
                st.warning("No conversations match the selected filters.")
        else:
            st.warning("No conversations found in the specified directory.")
            st.info("If you've run the pipeline, check that the conversations were generated correctly.")
    
    elif page == "Knowledge Units":
        st.header("Extracted Knowledge Units")
        
        # Load knowledge units
        knowledge_dir = os.path.join(output_dir, "knowledge_units")
        knowledge_units = load_knowledge_units(knowledge_dir)
        
        if knowledge_units:
            # Flatten the knowledge units if they're nested
            flat_knowledge_units = []
            for ku in knowledge_units:
                if isinstance(ku, list):
                    flat_knowledge_units.extend(ku)
                else:
                    flat_knowledge_units.append(ku)
            
            # Create a dataframe for filtering
            knowledge_df = pd.DataFrame([
                {
                    "ID": i,
                    "Source": ku.get('source', 'Unknown') if isinstance(ku, dict) else 'Unknown',
                    "Type": ku.get('type', 'Unknown').title() if isinstance(ku, dict) else 'Unknown',
                    "Confidence": float(ku.get('confidence', 0)) if isinstance(ku, dict) else 0.0,
                    "Iteration": int(ku.get('iteration', 0)) if isinstance(ku, dict) else 0,
                    "Content": ku.get('content', '')[:100] + '...' if isinstance(ku, dict) and len(ku.get('content', '')) > 100 else ku.get('content', '')
                }
                for i, ku in enumerate(flat_knowledge_units)
                if isinstance(ku, dict)
            ])
            
            # Add filters in a sidebar-like column
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Filters")
                
                # Filter by knowledge type
                selected_types = st.multiselect(
                    "Knowledge type",
                    options=sorted(knowledge_df['Type'].unique()),
                    default=sorted(knowledge_df['Type'].unique())
                )
                
                # Filter by iteration
                iterations = sorted(knowledge_df['Iteration'].unique())
                selected_iterations = st.multiselect(
                    "Iteration",
                    options=iterations,
                    default=iterations
                )
                
                # Filter by confidence
                min_confidence = st.slider(
                    "Minimum confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
                
                # Search by content
                search_query = st.text_input("Search in content", "")
                
                # Sort options
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Confidence (high to low)", "Confidence (low to high)", "Iteration (newest first)", "Iteration (oldest first)"]
                )
                
                # Display options
                display_mode = st.radio(
                    "Display mode",
                    options=["Cards", "Table"]
                )
            
            # Apply filters
            filtered_df = knowledge_df[
                (knowledge_df['Type'].isin(selected_types)) &
                (knowledge_df['Iteration'].isin(selected_iterations)) &
                (knowledge_df['Confidence'] >= min_confidence)
            ]
            
            # Apply search filter if provided
            if search_query:
                filtered_df = filtered_df[filtered_df['Content'].str.contains(search_query, case=False, na=False)]
            
            # Apply sorting
            if sort_by == "Confidence (high to low)":
                filtered_df = filtered_df.sort_values(by='Confidence', ascending=False)
            elif sort_by == "Confidence (low to high)":
                filtered_df = filtered_df.sort_values(by='Confidence', ascending=True)
            elif sort_by == "Iteration (newest first)":
                filtered_df = filtered_df.sort_values(by='Iteration', ascending=False)
            elif sort_by == "Iteration (oldest first)":
                filtered_df = filtered_df.sort_values(by='Iteration', ascending=True)
            
            with col2:
                # Display filtered knowledge units
                if not filtered_df.empty:
                    st.subheader(f"Knowledge Units ({len(filtered_df)} of {len(knowledge_df)})")
                    
                    if display_mode == "Table":
                        # Display as table
                        display_cols = ["Type", "Confidence", "Iteration", "Content"]
                        st.dataframe(
                            filtered_df[display_cols],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Allow selection of a knowledge unit to view details
                        selected_ku_idx = st.selectbox(
                            "Select a knowledge unit to view details",
                            options=filtered_df.index,
                            format_func=lambda x: f"{filtered_df.loc[x, 'Type']} (Conf: {filtered_df.loc[x, 'Confidence']:.2f}) - {filtered_df.loc[x, 'Content']}"
                        )
                        
                        if selected_ku_idx is not None:
                            ku = knowledge_units[filtered_df.loc[selected_ku_idx, 'ID']]
                            st.subheader("Knowledge Unit Details")
                            display_knowledge_unit(ku)
                    else:
                        # Display as cards
                        for _, row in filtered_df.iterrows():
                            ku = knowledge_units[row['ID']]
                            display_knowledge_unit(ku)
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
            # Create tabs for different metric views
            tab1, tab2, tab3 = st.tabs(["Evolution Charts", "Detailed Metrics", "Insights"])
            
            with tab1:
                # Plot metrics over time
                st.subheader("Graph Evolution")
                plot_metrics_over_time(metrics)
                
                # Add explanation
                with st.expander("Understanding these metrics"):
                    st.markdown("""
                    **Graph Growth**: Shows how the number of nodes (personas, concepts) and edges (relationships) increases over iterations.
                    
                    **Graph Properties**:
                    - **Density**: Ratio of actual connections to possible connections. Higher values indicate a more interconnected graph.
                    - **Average Clustering**: Measures how nodes tend to cluster together. Higher values suggest the formation of tight-knit communities.
                    
                    **Community Structure**:
                    - **Modularity**: Measures the strength of division into communities. Higher values indicate well-defined communities.
                    - **Communities**: The number of distinct communities detected in the graph.
                    """)
            
            with tab2:
                # Display detailed metrics for a selected iteration
                st.subheader("Detailed Metrics by Iteration")
                
                # Create a selectbox for iteration
                iterations = sorted(list(set([m.get('iteration', 0) for m in metrics])))
                selected_iteration = st.selectbox(
                    "Select iteration",
                    options=iterations,
                    index=len(iterations)-1  # Default to latest iteration
                )
                
                # Get metrics for selected iteration
                selected_metric = next((m for m in metrics if m.get('iteration') == selected_iteration), None)
                
                if selected_metric:
                    # Create a dataframe for the metrics
                    metrics_items = {k: v for k, v in selected_metric.items() if k != 'iteration' and k != 'timestamp'}
                    
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
                        
                        # Format the key for better display
                        formatted_key = k.replace('_', ' ').title()
                        
                        formatted_items.append((formatted_key, formatted_value))
                    
                    # Group metrics by category
                    basic_metrics = [item for item in formatted_items if item[0].startswith(('Num', 'Density', 'Avg'))]
                    centrality_metrics = [item for item in formatted_items if 'Centrality' in item[0]]
                    community_metrics = [item for item in formatted_items if any(term in item[0] for term in ['Community', 'Modularity', 'Cluster'])]
                    other_metrics = [item for item in formatted_items if item not in basic_metrics + centrality_metrics + community_metrics]
                    
                    # Create columns for different metric categories
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Basic Metrics")
                        basic_df = pd.DataFrame(basic_metrics, columns=["Metric", "Value"])
                        basic_df['Display_Value'] = basic_df['Value'].apply(
                            lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
                        )
                        st.table(basic_df[["Metric", "Display_Value"]].rename(columns={"Display_Value": "Value"}))
                        
                        st.subheader("Community Metrics")
                        community_df = pd.DataFrame(community_metrics, columns=["Metric", "Value"])
                        community_df['Display_Value'] = community_df['Value'].apply(
                            lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
                        )
                        st.table(community_df[["Metric", "Display_Value"]].rename(columns={"Display_Value": "Value"}))
                    
                    with col2:
                        st.subheader("Centrality Metrics")
                        centrality_df = pd.DataFrame(centrality_metrics, columns=["Metric", "Value"])
                        centrality_df['Display_Value'] = centrality_df['Value'].apply(
                            lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
                        )
                        st.table(centrality_df[["Metric", "Display_Value"]].rename(columns={"Display_Value": "Value"}))
                        
                        if other_metrics:
                            st.subheader("Other Metrics")
                            other_df = pd.DataFrame(other_metrics, columns=["Metric", "Value"])
                            other_df['Display_Value'] = other_df['Value'].apply(
                                lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
                            )
                            st.table(other_df[["Metric", "Display_Value"]].rename(columns={"Display_Value": "Value"}))
                else:
                    st.warning(f"No metrics found for iteration {selected_iteration}")
            
            with tab3:
                st.subheader("Graph Insights")
                
                # Get the latest metrics
                latest_metric = max(metrics, key=lambda x: x.get('iteration', 0))
                latest_iteration = latest_metric.get('iteration', 0)
                
                st.markdown(f"### Key Insights from Iteration {latest_iteration}")
                
                # Extract key metrics
                num_nodes = latest_metric.get('num_nodes', 0)
                num_edges = latest_metric.get('num_edges', 0)
                density = latest_metric.get('density', 0)
                avg_clustering = latest_metric.get('avg_clustering', 0)
                modularity = latest_metric.get('modularity', 0)
                
                # Calculate growth rates if we have more than one iteration
                if len(metrics) > 1:
                    previous_metrics = [m for m in metrics if m.get('iteration', 0) == latest_iteration - 1]
                    if previous_metrics:
                        prev_metric = previous_metrics[0]
                        node_growth = (num_nodes - prev_metric.get('num_nodes', 0)) / prev_metric.get('num_nodes', 1) * 100
                        edge_growth = (num_edges - prev_metric.get('num_edges', 0)) / prev_metric.get('num_edges', 1) * 100
                        
                        st.markdown(f"""
                        #### Growth
                        - The knowledge graph grew by **{node_growth:.1f}%** in nodes and **{edge_growth:.1f}%** in edges in the last iteration.
                        """)
                
                # Display insights based on metrics
                st.markdown(f"""
                #### Structure
                - The graph contains **{num_nodes}** nodes and **{num_edges}** edges.
                - Graph density is **{density:.4f}**, indicating a {'highly' if density > 0.5 else 'moderately' if density > 0.2 else 'sparsely'} connected network.
                - Average clustering coefficient is **{avg_clustering:.4f}**, suggesting {'strong' if avg_clustering > 0.6 else 'moderate' if avg_clustering > 0.3 else 'weak'} community formation.
                """)
                
                if 'modularity' in latest_metric:
                    st.markdown(f"""
                    #### Communities
                    - Modularity score is **{modularity:.4f}**, indicating {'well-defined' if modularity > 0.5 else 'somewhat defined' if modularity > 0.3 else 'poorly defined'} community structure.
                    - The graph contains **{latest_metric.get('num_communities', 0)}** distinct communities.
                    """)
                
                # Add recommendations based on metrics
                st.markdown("""
                #### Recommendations
                """)
                
                recommendations = []
                
                if density < 0.1:
                    recommendations.append("- Consider generating more conversations to increase connectivity between concepts and personas.")
                
                if avg_clustering < 0.3:
                    recommendations.append("- The low clustering suggests weak community formation. Try focusing conversations on related topics to strengthen communities.")
                
                if 'modularity' in latest_metric and latest_metric.get('modularity', 0) < 0.3:
                    recommendations.append("- Community structure is weak. Consider focusing on more specialized topics within each iteration.")
                
                if not recommendations:
                    recommendations.append("- The graph shows healthy growth and structure. Continue with the current pipeline configuration.")
                
                for rec in recommendations:
                    st.markdown(rec)
        else:
            st.warning("No metrics data found in the specified directory.")
            st.info("If you've run the pipeline, check that the graph analysis was completed correctly.")

if __name__ == "__main__":
    main() 