import spacy
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import math

logging.basicConfig(level=logging.DEBUG, format= (
    "%(asctime)s | %(levelname)s | "
    "%(filename)s:%(lineno)d | "
    "%(funcName)s | %(message)s "
))
                    
logger=logging.getLogger(__name__)

model=SentenceTransformer("all-MiniLM-L6-v2")
encoder_model=CrossEncoder('cross-encoder/nli-deberta-v3-base')

nlp=spacy.load("en_core_web_sm")

def keyword_extractor(text :str):
    doc=nlp(text.lower())
    keywords=set()
    try:
        for words in doc:
            if words.pos_ in {'NOUN', 'ADJ', 'VERB'}:
                if not words.is_stop and not words.is_punct:
                    keywords.add(words.lemma_)
    except Exception:
        logger.exception("Keyword extractor failed")
        raise
    return keywords

def keyword_score(customer_text, agent_text):
    customer_keywords=keyword_extractor(customer_text)
    agent_keywords=keyword_extractor(agent_text)

    matched_words=customer_keywords.intersection(agent_keywords)
    if len(customer_keywords) == 0:
        matched_score = 0.0
    else:
        matched_score = len(matched_words) / len(customer_keywords)
    return matched_score

def Paraphrasing_check(customer_text, agent_text):
    try:
        entailment_score=encoder_model(agent_text, customer_text)
    except Exception:
        logger.exception("Paraphrasing failed")
        raise
    return entailment_score


def similarity_score(customer_list:list, agent_list:list):
    semantic_score_sum=0
    entailment_score=[]
    count=[]
    n=0
    for i, texts in enumerate(customer_list):
        if i==0 or i==len(customer_list)-1:
            continue
        text1=customer_list[i-1].get('text')+customer_list[i].get('text')+customer_list[i+1].get('text')
        text2=agent_list[i-1].get('text')+agent_list[i].get('text')+agent_list[i+1].get('text')
        
        embeddings1=model.encode(text1, normalize_embeddings=True)
        embeddings2=model.encode(text2, normalize_embeddings=True)

        

        score=cosine_similarity(embeddings1, embeddings2)[0][0]
        count.append(score)

# adding the weight value of the semantic score
# more recent conversation will have more importance in overall conversation 
# taking index of the semantic score in the list as the weight

    weighted_sum=0
    total_weight=0
    for i, c in enumerate(count):
        weighted_sum+=(i+1)*c 
        total_weight+=(i+1)

    if total_weight == 0:
        weight_semantic_score = 0.0
    else:
        weight_semantic_score=weighted_sum/total_weight
    
    # Normalize cosine similarity from [-1, 1] to [0, 1] range
    # Using formula: (score + 1) / 2
    normalized_score = (weight_semantic_score+1)/2

    return round(normalized_score, 2)

#takingg mean of both the values

def overall_attention(similarity_score, keyword_score):
    return (similarity_score+keyword_score)/2

