"""
Utility functions for the hybrid medical chatbot
"""
import json
import pickle
import numpy as np
from pathlib import Path


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj, filepath):
    """Save object as pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


def load_intents(filepath=None):
    """Load intents from JSON file"""
    if filepath is None:
        filepath = get_project_root() / 'data' / 'intents.json'
    return load_json(filepath)


def prepare_data_from_intents(intents_data):
    """
    Extract patterns and tags from intents
    Returns: patterns (list), tags (list)
    """
    patterns = []
    tags = []
    
    for intent in intents_data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(tag)
    
    return patterns, tags


def get_response(tag, intents_data):
    """
    Get a random response for a given tag
    """
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "I'm not sure how to help with that. Please consult a doctor."


if __name__ == "__main__":
    # Test functions
    intents = load_intents()
    print(f"Loaded {len(intents['intents'])} intents")
    
    patterns, tags = prepare_data_from_intents(intents)
    print(f"Total patterns: {len(patterns)}")
    print(f"Unique tags: {len(set(tags))}")