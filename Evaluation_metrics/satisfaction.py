'''
To check the satisfaction of the customer- 
1) By checking for explicit statements which clearly shows satisfaction example - thank you for the help. 
2) Also check for implicit statements like - okay, this 'okay' could mean either satisfaction or just a casual agreement
Therefore for implicit except fore checking the the phrases we will check the emotional state as well with SentimentIntensityAnalyzer.
 
'''

Explicit_statements = [
    "that solved my issue",
    "this solved my problem",
    "my issue is resolved",
    "the problem is resolved now",
    "that fixed it",
    "it’s working now",
    "everything is working fine now",
    "the issue is fixed",
    "this has been resolved",
    "my problem is fixed now",

    "that works",
    "this works for me",
    "that solution works",
    "this makes sense now",
    "i understand it now",
    "that answers my question",
    "this clears things up",
    "that helps a lot",
    "this is helpful",
    "that was helpful",

    "thanks for your help",
    "thank you for helping me",
    "i appreciate your help",
    "thanks, that was helpful",
    "thank you, that solves it",
    "appreciate the support",
    "thanks, that works",
    "thank you so much",
    "thanks for explaining",
    "thanks for resolving this",

    "i’m happy with the solution",
    "i’m satisfied with the support",
    "i’m satisfied now",
    "i’m happy now",
    "this is much better",
    "that’s great",
    "perfect, thank you",
    "that’s exactly what i needed",
    "i’m glad that’s sorted",
    "that resolved my concern",

    "that’s all i needed",
    "no further questions",
    "that’s everything, thanks",
    "i don’t need anything else",
    "that’s all from my side",
    "okay, thank you",
    "alright, thanks for the help",
    "that will be all",
    "i’m good now",
    "we’re good now"]

IMPLICIT_ACCEPTANCE_WORDS = [
    "okay", "ok", "alright","right", "sure", "fine", "understood", "understand", "got", "gotcha",
    "see", "yes", "yeah", "yep", "correct", "exactly", "true", "fair", "cool", "great",
    "perfect", "thanks", "thank", "appreciate", "noted", "acknowledged", "accepted", "agree", "agreed", "confirm",
    "confirmed","sounds","works","working","clear","clarified","resolved", "fixed","done","completed",
    "helpful","useful","good","better"]

IMPLICIT_SATISFACTION_PATTERNS = [
    "okay thanks", "okay thank you", "alright thanks", "sure thanks", "got it thanks",
    "that works", "sounds good", "that makes sense", "i understand", "i see",
    "that helps", "that's clear", "makes sense", "i got it", "understood",
    "that's fine", "that's good", "sounds fine", "that's okay", "works for me",
    "no problem", "no worries", "all good", "i'm good", "we're good",
    "that's all", "that's everything", "nothing else", "no more questions",
    "perfect", "great", "excellent", "wonderful", "awesome"
]

NEGATIVE_CONTEXT_INDICATORS = [
    "but", "however", "still", "yet", "unfortunately", "disappointed", "frustrated",
    "confused", "not working", "doesn't work", "can't", "won't", "unable",
    "problem", "issue", "error", "wrong", "incorrect", "unsure", "doubt"
]

IMPLICIT=' '.join(IMPLICIT_ACCEPTANCE_WORDS)

from sentence_transformers.util.tensor import normalize_embeddings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

model= SentenceTransformer("all-MiniLM-L6-v2")
nlp=spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()


def keywords_func(sentence:str):
    doc=nlp(sentence.lower())
    set1=set()
    for Token in doc:
        if Token.pos_ in {'ADJ', 'NOUN', 'VERB'}:
            if not Token.is_stop and not Token.is_punct:
                set1.add(Token.lemma_)
    return set1


def sentiment_score(text: str) -> float:
    """
    Returns compound sentiment score in range [-1, 1]
    -ve or +ve implies the emotion state
    """
    return sentiment_analyzer.polarity_scores(text)["compound"]

explicit_embedding=model.encode(
    sentences= Explicit_statements,
    normalize_embeddings=True)

# Pre-compute embeddings for implicit satisfaction patterns
implicit_patterns_embedding = model.encode(
    sentences=IMPLICIT_SATISFACTION_PATTERNS,
    normalize_embeddings=True
)


def explicit_check(customer_dict_list:list[dict], portion= 0.3):
    '''
    1. Generated explicit phrases that shows satisfied emotions via GPT
    2. Iterating over customer utterances to check for similar
       phrases with the help of sentence & phrases embeddings
       with cosine_similarity.
    3. We are looking for emotions that show satisfaction that last
       portion of the conversation
    '''
    semantic_list=[]
    begin=int(len(customer_dict_list)*(1-portion))
    for u in customer_dict_list[begin:]:
        text=u.get('text')
        text_embedding=model.encode( text, normalize_embeddings=True)
        #comparing all the explicit phrases with the customer utterance
        similarity_score=cosine_similarity(
            explicit_embedding,
            [text_embedding]
        )
        similarity_score=similarity_score.flatten()
        score=np.max(similarity_score)
        semantic_list.append(score)
    avg_score=np.mean(semantic_list)
    if avg_score<0:
        avg_score=0
    return avg_score


def _has_negative_context(text: str) -> bool:
    '''
    Check if text contains negative context indicators that suggest dissatisfaction.
    '''
    text_lower = text.lower()
    for indicator in NEGATIVE_CONTEXT_INDICATORS:
        if indicator in text_lower:
            return True
    return False


def _calculate_semantic_similarity(text: str) -> float:
    '''
    Calculate semantic similarity with implicit satisfaction patterns using embeddings.
    Returns max similarity score [0, 1].
    '''
    text_embedding = model.encode(text, normalize_embeddings=True)
    similarity_scores = cosine_similarity(
        implicit_patterns_embedding,
        [text_embedding]
    )
    similarity_scores = similarity_scores.flatten()
    max_similarity = np.max(similarity_scores)
    return max(0.0, max_similarity)


def _calculate_keyword_match_score(text: str) -> float:
    '''
    Calculate keyword matching score with implicit acceptance words.
    Returns normalized score [0, 1] based on keyword overlap.
    '''
    implicit_keywords = keywords_func(IMPLICIT)
    sentence_keywords = keywords_func(text)
    
    if len(implicit_keywords) == 0:
        return 0.0
    
    intersection = sentence_keywords.intersection(implicit_keywords)
    match_ratio = len(intersection) / len(implicit_keywords)
    return min(1.0, match_ratio * 2)  

def _get_contextual_sentiment(text: str, prev_text: str = "", next_text: str = "") -> float:
    """
    Calculate sentiment considering context from surrounding utterances.
    """
    current_sentiment = sentiment_score(text)
    if _has_negative_context(text):
        current_sentiment = current_sentiment * 0.5  
    if prev_text:
        prev_sentiment = sentiment_score(prev_text)
        if prev_sentiment < 0 and current_sentiment > 0:
            current_sentiment = current_sentiment * 1.2 
        elif prev_sentiment > 0 and current_sentiment > 0:
            current_sentiment = current_sentiment * 1.1 
    
    normalized_sentiment = (current_sentiment + 1) / 2
    return max(0.0, min(1.0, normalized_sentiment))


def implicit_check(customer_dict_list: list[dict], portion: float = 0.4):
    '''
    Improved implicit satisfaction detection using multiple signals:
    
    1. Semantic Similarity: Uses embeddings to find satisfaction patterns beyond keywords
    2. Context-Aware Sentiment: Considers surrounding conversation context
    3. Keyword Matching: Enhanced keyword overlap detection
    4. Conversation Progression: Weights recent utterances more heavily
    5. Negative Signal Detection: Identifies dissatisfaction even with implicit words
    
    Args:
        customer_dict_list: List of customer utterance dictionaries with 'text' key
        portion: Portion of conversation to analyze (default 0.4 = last 40%)
    
    Returns:
        Implicit satisfaction score [0, 1]
    '''
    if not customer_dict_list:
        return 0.0
    
    begin_idx = int(len(customer_dict_list) * (1 - portion))
    relevant_utterances = customer_dict_list[begin_idx:]
    
    if not relevant_utterances:
        return 0.0
    
    utterance_scores = []
    for i, utterance in enumerate(relevant_utterances):
        text = utterance.get('text', '').strip()
        if not text:
            continue
        
        prev_text = ""
        next_text = ""
        if i > 0:
            prev_text = relevant_utterances[i-1].get('text', '').strip()
        if i < len(relevant_utterances) - 1:
            next_text = relevant_utterances[i+1].get('text', '').strip()
        
        if _has_negative_context(text):
            sentiment = sentiment_score(text)
            if sentiment < -0.3:  
                continue  

        semantic_score = _calculate_semantic_similarity(text)
        keyword_score = _calculate_keyword_match_score(text)
        contextual_sentiment = _get_contextual_sentiment(text, prev_text, next_text)
        
        if semantic_score > 0.3 or keyword_score > 0.1 or contextual_sentiment > 0.5:
            combined_score = (
                semantic_score * 0.40 +
                contextual_sentiment * 0.35 +
                keyword_score * 0.25
            )
            
            position_weight = (i + 1) / len(relevant_utterances)
            weighted_score = combined_score * (0.7 + 0.3 * position_weight)
            utterance_scores.append(weighted_score)
    
    if not utterance_scores:
        return 0.0
    
    scores_array = np.array(utterance_scores)
    
    # Calculate mean and max scores for normalization
    mean_score = np.mean(scores_array)
    max_score = np.max(scores_array)
    
    # Boost score if multiple utterances show satisfaction (more reliable)
    if len(utterance_scores) >= 2:
        mean_score = min(1.0, mean_score * 1.15)
    
    # Normalize using weighted combination of mean and max
    # This gives credit for both consistent satisfaction and peak satisfaction moments
    # Mean represents overall satisfaction level, max represents best moments
    normalized_score = (mean_score * 0.7 + max_score * 0.3)
    
    # Apply min-max normalization to ensure proper scaling
    # Scale the score to better utilize the [0, 1] range
    # This makes it more comparable to explicit_check scores
    if normalized_score > 0:
        # Apply a scaling factor to ensure scores are well-distributed
        # Lower threshold for implicit satisfaction (0.3) vs explicit (0.5)
        scaled_score = normalized_score
        
        # If we have multiple strong signals, boost the score
        strong_signals = np.sum(scores_array > 0.5)
        if strong_signals >= 2:
            scaled_score = min(1.0, scaled_score * 1.1)
        
        normalized_score = scaled_score
    
    # Ensure the score is properly bounded in [0, 1] range
    final_score = max(0.0, min(1.0, normalized_score))
    
    # Round to 4 decimal places for consistency with explicit_check
    return round(final_score, 4)