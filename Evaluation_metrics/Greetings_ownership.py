CANONICAL_GREETINGS = [
    "Hello, how may I help you?",
    "Hi, how can I assist you today?",
    "Good morning, thank you for contacting support",
    "Good afternoon, how may I help?",
    "Good evening, how can I help you?",
    "Welcome to customer support",
    "Thank you for calling customer service",
    "Hello and welcome",
    "Hi there, how may I assist?",
    "Thanks for reaching out to us",
    "Thank you for contacting us today",
    "Hello, thank you for getting in touch",
    "Welcome, how may I assist you?",
    "Hi, thanks for calling",
    "Good morning, how can I assist you today?",
    "Good afternoon, thank you for reaching out",
    "Good evening, thank you for contacting us",
    "Hello, I’ll be happy to assist you",
    "Hi, I’m here to help you",
    "Welcome, thank you for calling",
    "Hello, how can I support you?",
    "Thank you for reaching customer support",
    "Hi, how may I help you today?",
    "Good morning and welcome",
    "Good afternoon and welcome",
    "Good evening and welcome",
    "Hello, thanks for contacting support",
    "Hi, thank you for reaching out to support",
    "Welcome to our support team",
    "Thank you for calling, how may I help?"
]


CANONICAL_OWNERSHIP_STRONG = [
    "I will take full responsibility for this issue",
    "I will personally handle this for you",
    "I will take care of this matter",
    "I will ensure this gets resolved",
    "I will make sure this is fixed",
    "I am responsible for resolving this issue",
    "I will see this through until it is resolved",
    "I will own this issue",
    "I will personally follow up on this",
    "I will handle this end to end"
]
CANONICAL_OWNERSHIP_ACTION = [
    "I will check this for you",
    "Let me look into this for you",
    "I will investigate this issue",
    "I will work on resolving this",
    "I will get this checked immediately",
    "I will review this right away",
    "I am checking this now",
    "Let me verify the details for you",
    "I will examine what went wrong",
    "I will look into the cause of this issue"
]
CANONICAL_OWNERSHIP_SUPPORT = [
    "I understand your concern and will help you",
    "I understand the issue you are facing",
    "I can help you with this",
    "Let me help you resolve this",
    "I am here to assist you with this",
    "I see why this is frustrating",
    "I understand how this impacts you",
    "I will assist you with this issue",
    "I am here to support you",
    "I understand your issue and will help"
]

CANONICAL_OWNERSHIP = (
    CANONICAL_OWNERSHIP_STRONG +
    CANONICAL_OWNERSHIP_ACTION +
    CANONICAL_OWNERSHIP_SUPPORT
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model= SentenceTransformer("all-MiniLM-L6-v2")

greetings_embeddings=model.encode(
    sentences=CANONICAL_GREETINGS,
    normalize_embeddings=True
)

def check_greetings(
    agent_list:list[dict])-> int:
    final_value=0
    for i, line in enumerate(agent_list):
        if i<3: #checking if the agent greeted in the first 5 lines
            sentence_embedding=model.encode(
                sentences=line.get('text'),
                normalize_embeddings=True
            )
            similarity_matrix=cosine_similarity(
                [sentence_embedding], #(1,384)
                greetings_embeddings  #(3,384)
            ) 
            similarity_matrix=similarity_matrix.flatten()
            max_value=np.max(similarity_matrix)
            if max_value>0.65:
                final_value=1
                break

            #Let's say greeting_embeddings is for 3 sentences so the greeting embeddings will have the shape (3, 384)
            #and the sentence embedding will have a shape (384,) so after [sentence_embedding] it will be (1, 384)\
            # cosine similarity will be of shape (1, 3) eg. [0.45, 0.78, 0,12] 
    
    return final_value

ownership_embeddings=model.encode(
    sentences=CANONICAL_OWNERSHIP,
    normalize_embeddings=True
)

def check_ownership(agent_list:list[dict])-> float:
    if not agent_list:
        return 0.0
    
    all_scores = []
    
    for line in agent_list:
        sentence_embedding = model.encode(
            sentences=line.get('text'),
            normalize_embeddings=True
        )
        similarity_matrix = cosine_similarity(
            [sentence_embedding],
            ownership_embeddings
        )
        
        # Get average similarity for this utterance
        utterance_score = np.mean(similarity_matrix)
        all_scores.append(utterance_score)
    
    if not all_scores:
        return 0.0
    
    average_score = np.mean(all_scores)
    
    # Normalize from [-1, 1] to [0, 1] and ensure bounds
    normalized_score = (average_score + 1) / 2
    return max(0.0, min(1.0, normalized_score))




