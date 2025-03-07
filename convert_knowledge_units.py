#!/usr/bin/env python3
"""
Convert knowledge units from pipeline_output to the format needed for the Streamlit app
and copy them to pipeline_demo_output.
"""

import os
import json
import glob
import uuid
from pathlib import Path

def convert_knowledge_unit(knowledge_unit, source_file, iteration):
    """
    Convert a knowledge unit from the pipeline format to the format expected by the Streamlit app.
    """
    # Extract concepts
    concepts = knowledge_unit.get('concepts', [])
    
    # Extract relationships
    relationships = knowledge_unit.get('relationships', [])
    
    # Create new knowledge units (one per concept and relationship)
    new_knowledge_units = []
    
    # Process concepts
    for concept in concepts:
        new_ku = {
            'id': str(uuid.uuid4()),
            'type': 'concept',
            'content': concept.get('description', ''),
            'subject': concept.get('name', ''),
            'confidence': 0.85,  # Default confidence
            'source': os.path.basename(source_file),
            'iteration': iteration,
            'domain': concept.get('domain', 'General')
        }
        new_knowledge_units.append(new_ku)
    
    # Process relationships
    for relationship in relationships:
        new_ku = {
            'id': str(uuid.uuid4()),
            'type': 'relationship',
            'content': relationship.get('description', ''),
            'subject': relationship.get('source', ''),
            'predicate': relationship.get('type', ''),
            'object': relationship.get('target', ''),
            'confidence': 0.75,  # Default confidence
            'source': os.path.basename(source_file),
            'iteration': iteration
        }
        new_knowledge_units.append(new_ku)
    
    # If no concepts or relationships, create a generic knowledge unit
    if not new_knowledge_units:
        new_ku = {
            'id': str(uuid.uuid4()),
            'type': 'insight',
            'content': 'Generic insight from conversation',
            'confidence': 0.6,
            'source': os.path.basename(source_file),
            'iteration': iteration
        }
        new_knowledge_units.append(new_ku)
    
    return new_knowledge_units

def process_knowledge_units():
    """
    Process all knowledge units from pipeline_output and save them to pipeline_demo_output.
    """
    # Create output directory if it doesn't exist
    os.makedirs('pipeline_demo_output/knowledge_units', exist_ok=True)
    
    # Process individual knowledge unit files
    individual_files = glob.glob('pipeline_output/knowledge_units/knowledge_*.json')
    
    # Group knowledge units by iteration
    knowledge_units_by_iteration = {}
    all_knowledge_units = []
    
    for file_path in individual_files:
        try:
            with open(file_path, 'r') as f:
                knowledge_unit = json.load(f)
            
            # Extract iteration from filename
            filename = os.path.basename(file_path)
            if '_' in filename:
                parts = filename.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    iteration = int(parts[1])
                else:
                    iteration = 0
            else:
                iteration = 0
            
            # Convert the knowledge unit
            new_knowledge_units = convert_knowledge_unit(knowledge_unit, file_path, iteration)
            
            # Add to the appropriate iteration group
            if iteration not in knowledge_units_by_iteration:
                knowledge_units_by_iteration[iteration] = []
            
            knowledge_units_by_iteration[iteration].extend(new_knowledge_units)
            all_knowledge_units.extend(new_knowledge_units)
            
            # Save individual knowledge unit
            output_path = os.path.join('pipeline_demo_output/knowledge_units', filename)
            with open(output_path, 'w') as f:
                json.dump(new_knowledge_units, f, indent=2)
                
            print(f"Converted and saved: {filename} with {len(new_knowledge_units)} knowledge units")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save iteration files
    for iteration, knowledge_units in knowledge_units_by_iteration.items():
        output_path = f'pipeline_demo_output/knowledge_units/knowledge_units_iteration_{iteration}.json'
        with open(output_path, 'w') as f:
            json.dump(knowledge_units, f, indent=2)
        
        print(f"Saved iteration file: knowledge_units_iteration_{iteration}.json with {len(knowledge_units)} knowledge units")
    
    # Save all knowledge units
    output_path = 'pipeline_demo_output/knowledge_units/all_knowledge_units.json'
    with open(output_path, 'w') as f:
        json.dump(all_knowledge_units, f, indent=2)
    
    print(f"Saved all knowledge units: all_knowledge_units.json with {len(all_knowledge_units)} knowledge units")

if __name__ == "__main__":
    print("Converting knowledge units from pipeline_output to pipeline_demo_output...")
    process_knowledge_units()
    print("Conversion complete!") 