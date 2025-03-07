"""
PersonaHub Elite Personas + Agentic Deep Graph Reasoning Pipeline

This package implements a graph-based reasoning pipeline that leverages elite personas
from PersonaHub to create self-organizing knowledge networks through iterative
conversation generation and knowledge extraction.

Key components:
- Knowledge Graph: Represents personas, concepts, and their relationships
- Conversation Generator: Creates conversations between personas
- Knowledge Extractor: Extracts structured knowledge from conversations
- Graph Updater: Updates the knowledge graph with new insights
- Pipeline: Orchestrates the iterative reasoning process
"""

__version__ = "0.1.0" 

# Expose modules for easier importing
from . import knowledge_graph
from . import conversation_generator
from . import knowledge_extractor
from . import graph_updater
from . import pipeline 