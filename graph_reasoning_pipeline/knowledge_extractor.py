import json
import os
import logging
import openai
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_extraction_prompt(conversation: Dict[str, Any]) -> str:
    """
    Create a prompt for extracting knowledge from a conversation.
    
    Args:
        conversation: Conversation dictionary
        
    Returns:
        Extraction prompt for the LLM
    """
    content = conversation.get('content', '')
    
    prompt = f"""
    Extract the key knowledge units from this conversation:
    
    {content}
    
    Analyze the conversation and identify:
    
    1. Key concepts discussed (at least 3-5)
    2. Relationships between concepts (at least 2-3)
    3. Novel insights or facts generated (at least 2-3)
    
    Format your response as JSON with the following structure:
    {{
        "concepts": [
            {{
                "name": "concept_name",
                "description": "brief description",
                "domain": "domain_name"
            }}
        ],
        "relationships": [
            {{
                "source": "source_concept_name",
                "target": "target_concept_name",
                "type": "relationship_type",
                "description": "description of the relationship",
                "evidence": "quote from the conversation supporting this relationship"
            }}
        ],
        "insights": [
            {{
                "description": "description of the insight",
                "related_concepts": ["concept1", "concept2"],
                "novelty_score": 0.8,
                "evidence": "quote from the conversation supporting this insight"
            }}
        ]
    }}
    
    Ensure all JSON is properly formatted and valid. Use only concepts that are explicitly mentioned or strongly implied in the conversation.
    """
    
    return prompt

def query_gpt4o(prompt: str, max_retries: int = 3) -> str:
    """
    Query GPT-4o with the given prompt.
    
    Args:
        prompt: The prompt to send to GPT-4o
        max_retries: Maximum number of retries on failure
        
    Returns:
        The model's response
    """
    client = openai.OpenAI()
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.2,  # Lower temperature for more consistent extraction
                messages=[
                    {"role": "system", "content": "You are a knowledge extraction assistant. Extract structured information from conversations accurately."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.warning(f"Error querying GPT-4o (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
    
    return "Error extracting knowledge."

def parse_extraction_response(response: str) -> Dict[str, Any]:
    """
    Parse the extraction response from the LLM.
    
    Args:
        response: LLM response string
        
    Returns:
        Parsed knowledge units as a dictionary
    """
    # Try to extract JSON from the response
    try:
        # Find JSON content (it might be surrounded by markdown code blocks or other text)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            knowledge_units = json.loads(json_str)
            return knowledge_units
        else:
            logger.warning("No JSON found in the response")
            return {
                "concepts": [],
                "relationships": [],
                "insights": []
            }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from response: {e}")
        logger.debug(f"Response: {response}")
        return {
            "concepts": [],
            "relationships": [],
            "insights": []
        }

def extract_knowledge_from_conversation(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract knowledge units from a single conversation.
    
    Args:
        conversation: Conversation dictionary
        
    Returns:
        Extracted knowledge units
    """
    # Create extraction prompt
    prompt = create_extraction_prompt(conversation)
    
    # Query the model
    response = query_gpt4o(prompt)
    
    # Parse the response
    knowledge_units = parse_extraction_response(response)
    
    # Add metadata
    knowledge_units['source_conversation'] = conversation.get('personas', [])
    knowledge_units['extraction_timestamp'] = str(datetime.datetime.now())
    knowledge_units['conversation_iteration'] = conversation.get('iteration', 0)
    
    return knowledge_units

def extract_knowledge(conversations: List[Dict[str, Any]], 
                     output_dir: str = "knowledge_units") -> List[Dict[str, Any]]:
    """
    Extract knowledge from multiple conversations.
    
    Args:
        conversations: List of conversation dictionaries
        output_dir: Directory to save extracted knowledge
        
    Returns:
        List of extracted knowledge units
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    knowledge_units_list = []
    for i, conversation in enumerate(tqdm(conversations, desc="Extracting knowledge")):
        try:
            # Extract knowledge
            knowledge_units = extract_knowledge_from_conversation(conversation)
            
            # Add to list
            knowledge_units_list.append(knowledge_units)
            
            # Save individual knowledge units
            conversation_id = f"{conversation.get('iteration', 0)}_{i}"
            if 'personas' in conversation and len(conversation['personas']) >= 2:
                conversation_id = f"{conversation.get('iteration', 0)}_{conversation['personas'][0]}_{conversation['personas'][1]}"
            
            knowledge_file = os.path.join(
                output_dir, 
                f"knowledge_{conversation_id}.json"
            )
            with open(knowledge_file, 'w') as f:
                json.dump(knowledge_units, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error extracting knowledge from conversation {i}: {e}")
    
    # Save all knowledge units
    all_knowledge_file = os.path.join(output_dir, f"knowledge_units_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(all_knowledge_file, 'w') as f:
        json.dump(knowledge_units_list, f, indent=2)
    
    logger.info(f"Extracted knowledge from {len(knowledge_units_list)} conversations")
    return knowledge_units_list

def load_conversations(conversations_dir: str, iteration: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load conversations from files.
    
    Args:
        conversations_dir: Directory containing conversation files
        iteration: Specific iteration to load (None for all)
        
    Returns:
        List of conversation dictionaries
    """
    conversations = []
    
    # Find all conversation files
    for filename in os.listdir(conversations_dir):
        if not filename.endswith('.json'):
            continue
            
        if iteration is not None and not filename.startswith(f"conversation_{iteration}_"):
            continue
            
        file_path = os.path.join(conversations_dir, filename)
        try:
            with open(file_path, 'r') as f:
                conversation = json.load(f)
                conversations.append(conversation)
        except Exception as e:
            logger.warning(f"Error loading conversation from {file_path}: {e}")
    
    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations

if __name__ == "__main__":
    import datetime
    
    # Example usage
    conversations = load_conversations("conversations", iteration=0)
    knowledge_units = extract_knowledge(conversations)
    print(f"Extracted {len(knowledge_units)} knowledge units") 