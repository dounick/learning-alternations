from typing import List
import pickle
from numpy import random

class Node:
    def __init__(self, text: str, constituent: str, children: List['Node'], position: int, parent_position: int, pos: str):
        self.text = text
        self.constituent = constituent
        self.children = children
        self.position = position
        self.parent_position = parent_position
        self.pos = pos

    def __repr__(self):
        return f"Node(text='{self.text}', constituent='{self.constituent}', position={self.position}, parent_position={self.parent_position})"
    
def build_node_nosort(token, visited):
    if token in visited:
        return visited[token]
    
    # Recursively build children nodes first
    children = []
    for child in token.children:
        child_node = build_node_nosort(child, visited)
        children.append(child_node)
    
    left_children = sorted([c for c in children if c.position < token.i], key=lambda x: x.position)
    right_children = sorted([c for c in children if c.position > token.i], key=lambda x: x.position)
    
    # Build constituent
    parts = []
    for child in left_children:
        parts.append(child.constituent)
    parts.append(token.text)
    for child in right_children:
        parts.append(child.constituent)
    
    constituent = ' '.join(parts)
    
    # Create node (preserve original token.i for positions)
    node = Node(
        text=token.text,
        constituent=constituent,
        children=left_children + right_children,
        position=token.i,
        parent_position=token.head.i if token.head != token else -1,
        pos=token.pos_  # Store POS tag to identify punctuation
    )
    visited[token] = node
    return node

def build_node_random(token, visited):
    if token in visited:
        return visited[token]
    
    # Recursively build children nodes first
    children = []
    for child in token.children:
        child_node = build_node_random(child, visited)
        children.append(child_node)
    
    # Split children into punctuation and non-punctuation groups
    non_punct_children = [c for c in children if c.pos != "PUNCT"]
    non_punct_sorted_positions = sorted([c.position for c in non_punct_children])
    punct_children = [c for c in children if c.pos == "PUNCT"]
    
    # Sort non-punctuation children by constituent length (ascending)
    sorted_non_punct = sorted(
        non_punct_children,
        key=lambda x: len([word for word in x.constituent.split() if any(char.isalnum() for char in word)]),
        reverse=random.choice([True, False])
    )
    # reassign original positions to now sorted non-punct children    
    for i, child in enumerate(sorted_non_punct):
        child.position = non_punct_sorted_positions[i]
    
    # list of children, where non-punct children are sorted by constituent length
    children_sorted = non_punct_children + punct_children
    
    # Split into left/right groups based on original positions
    left_children = sorted([c for c in children_sorted if c.position < token.i], key=lambda x: x.position)
    right_children = sorted([c for c in children_sorted if c.position > token.i], key=lambda x: x.position)
    
    # Build constituent
    parts = []
    for child in left_children:
        parts.append(child.constituent)
    parts.append(token.text)
    for child in right_children:
        parts.append(child.constituent)
    
    constituent = ' '.join(parts)
    
    # Create node (preserve original token.i for positions)
    node = Node(
        text=token.text,
        constituent=constituent,
        children=left_children + right_children,
        position=token.i,
        parent_position=token.head.i if token.head != token else -1,
        pos=token.pos_  # Store POS tag to identify punctuation
    )
    visited[token] = node
    return node

def build_node(token, visited, short_first):
    if token in visited:
        return visited[token]
    
    # Recursively build children nodes first
    children = []
    for child in token.children:
        child_node = build_node(child, visited, short_first)
        children.append(child_node)
    
    # Split children into punctuation and non-punctuation groups
    non_punct_children = [c for c in children if c.pos != "PUNCT"]
    non_punct_sorted_positions = sorted([c.position for c in non_punct_children])
    punct_children = [c for c in children if c.pos == "PUNCT"]
    
    # Sort non-punctuation children by constituent length (ascending)
    sorted_non_punct = sorted(
        non_punct_children,
        key=lambda x: len([word for word in x.constituent.split() if any(char.isalnum() for char in word)]),
        reverse=not short_first
    )
    # reassign original positions to now sorted non-punct children    
    for i, child in enumerate(sorted_non_punct):
        child.position = non_punct_sorted_positions[i]
    
    # list of children, where non-punct children are sorted by constituent length
    children_sorted = non_punct_children + punct_children
    
    # Split into left/right groups based on original positions
    left_children = sorted([c for c in children_sorted if c.position < token.i], key=lambda x: x.position)
    right_children = sorted([c for c in children_sorted if c.position > token.i], key=lambda x: x.position)
    
    # Build constituent
    parts = []
    for child in left_children:
        parts.append(child.constituent)
    parts.append(token.text)
    for child in right_children:
        parts.append(child.constituent)
    
    constituent = ' '.join(parts)
    
    # Create node (preserve original token.i for positions)
    node = Node(
        text=token.text,
        constituent=constituent,
        children=left_children + right_children,
        position=token.i,
        parent_position=token.head.i if token.head != token else -1,
        pos=token.pos_  # Store POS tag to identify punctuation
    )
    visited[token] = node
    return node

def reorder_sentence(doc, short_first):
    root = [token for token in doc if token.head == token][0]
    visited = {}
    root_node = build_node(root, visited, short_first)
    return root_node.constituent

def reorder_sentence_random(doc):
    root = [token for token in doc if token.head == token][0]
    visited = {}
    root_node = build_node_random(root, visited)
    return root_node.constituent

def calculate_sorting_inversions(node):
    """
    Recursively calculate the number of inversions when sorting children.
    
    Args:
    node: The current node in the tree
    short_first: Whether sorting is done short-first (True) or long-first (False)
    
    Returns:
    A tuple (total_inversions, total_comparisons)
    """
    # Base case: leaf nodes or nodes with no children
    if not node.children:
        return 0, 0, 0
    
    # Non-punctuation children for comparison
    non_punct_children = [c for c in node.children if c.pos != "PUNCT"]
    
    # Calculate inversions for this node's children
    node_inversions_short = 0
    node_inversions_long = 0
    node_comparisons = 0
    
    if len(non_punct_children) > 1:
        # Calculate lengths, considering only alphanumeric words
        lengths = [
            len([word for word in child.constituent.split() if any(char.isalnum() for char in word)]) 
            for child in non_punct_children
        ]
        
        # Count inversions in the current node's children
        for i in range(len(lengths)):
            for j in range(i+1, len(lengths)):
                node_comparisons += 1
                if (lengths[i] > lengths[j]):
                    node_inversions_short += 1
                elif (lengths[i] < lengths[j]):
                    node_inversions_long += 1
    
    # Recursively process child nodes
    for child in node.children:
        child_inversions_short, child_inversions_long, child_comparisons = calculate_sorting_inversions(child)
        node_inversions_short += child_inversions_short
        node_inversions_long += child_inversions_long
        node_comparisons += child_comparisons
    
    return node_inversions_short, node_inversions_long, node_comparisons

def get_inversion_score(doc):
    """
    Calculate normalized inversion score for the entire document.
    
    Args:
    doc: The spaCy document
    short_first: Whether to calculate short-first (True) or long-first (False) inversions
    
    Returns:
    A dictionary with inversion metrics
    """
    # Find the root token
    root = [token for token in doc if token.head == token][0]
    visited = {}
    root_node = build_node_nosort(root, visited)
    
    # Calculate inversions
    node_inversions_short, node_inversions_long, node_comparisons = calculate_sorting_inversions(root_node)
        
    return {
        'short_inversions': node_inversions_short,
        'long_inversions': node_inversions_long,
        'total_comparisons': node_comparisons,
    }

def save_node_structure(node, filename):
    with open(filename, 'wb') as f:
        pickle.dump(node, f)

def load_node_structure(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)