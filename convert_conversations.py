#!/usr/bin/env python3
"""
Convert conversations from pipeline_output to the format needed for the Streamlit app
and copy them to pipeline_demo_output.
"""

import os
import json
import glob
import shutil
from pathlib import Path

def convert_conversation(conversation):
    """
    Convert a conversation from the pipeline format to the format expected by the Streamlit app.
    """
    # Extract the personas
    persona1 = conversation.get('personas', ['Unknown', 'Unknown'])[0]
    persona2 = conversation.get('personas', ['Unknown', 'Unknown'])[1]
    
    # Extract persona names from the descriptions
    persona1_name = persona1.replace('persona_', 'Persona ')
    persona2_name = persona2.replace('persona_', 'Persona ')
    
    # Extract the content and parse it into messages
    content = conversation.get('content', '')
    
    # Split the content by speaker indicators
    messages = []
    
    # Try to parse the conversation content into messages
    if '**' in content:
        # Format with ** markers for speakers
        parts = content.split('\n\n')
        for part in parts:
            if part.strip():
                if '**' in part:
                    speaker_end = part.find(':**')
                    if speaker_end > 0:
                        speaker = part[2:speaker_end].strip()
                        content = part[speaker_end+3:].strip()
                        messages.append({
                            'speaker': speaker,
                            'content': content
                        })
                else:
                    # If no speaker marker, use the last speaker
                    if messages:
                        messages.append({
                            'speaker': messages[-1]['speaker'],
                            'content': part
                        })
    else:
        # If no structured format, just add the whole content as one message
        messages.append({
            'speaker': 'System',
            'content': content
        })
    
    # Create the new conversation format
    new_conversation = {
        'persona1': persona1_name,
        'persona2': persona2_name,
        'topic': 'Conversation between experts',
        'iteration': conversation.get('iteration', 0),
        'messages': messages,
        'original_data': conversation  # Keep the original data for reference
    }
    
    return new_conversation

def process_conversations():
    """
    Process all conversations from pipeline_output and save them to pipeline_demo_output.
    """
    # Create output directory if it doesn't exist
    os.makedirs('pipeline_demo_output/conversations', exist_ok=True)
    
    # Process individual conversation files
    individual_files = glob.glob('pipeline_output/conversations/conversation_*.json')
    
    # Group conversations by iteration
    conversations_by_iteration = {}
    
    for file_path in individual_files:
        try:
            with open(file_path, 'r') as f:
                conversation = json.load(f)
                
            # Convert the conversation
            new_conversation = convert_conversation(conversation)
            
            # Add to the appropriate iteration group
            iteration = new_conversation['iteration']
            if iteration not in conversations_by_iteration:
                conversations_by_iteration[iteration] = []
            
            conversations_by_iteration[iteration].append(new_conversation)
            
            # Save individual conversation
            filename = os.path.basename(file_path)
            output_path = os.path.join('pipeline_demo_output/conversations', filename)
            with open(output_path, 'w') as f:
                json.dump(new_conversation, f, indent=2)
                
            print(f"Converted and saved: {filename}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save iteration files
    for iteration, conversations in conversations_by_iteration.items():
        output_path = f'pipeline_demo_output/conversations/conversations_iteration_{iteration}.json'
        with open(output_path, 'w') as f:
            json.dump(conversations, f, indent=2)
        
        print(f"Saved iteration file: conversations_iteration_{iteration}.json with {len(conversations)} conversations")

if __name__ == "__main__":
    print("Converting conversations from pipeline_output to pipeline_demo_output...")
    process_conversations()
    print("Conversion complete!") 