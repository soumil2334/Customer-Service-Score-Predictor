import math
from Evaluation_metrics.Attention import keyword_score, similarity_score, overall_attention
from Evaluation_metrics.Empathy import empathy_check
from Evaluation_metrics.Greetings_ownership import check_greetings, check_ownership
from Evaluation_metrics.Interruption import interuptions
from Evaluation_metrics.satisfaction import explicit_check, implicit_check
from Evaluation_metrics.Talk_to_listen import talk_to_listen

def Normalize_attention(customer_utterance_string, agent_utterance_string, customer_utterance_list, agent_utterance_list):
    '''
    Calculate attention metrics between customer and agent utterances.

    Args: customer_utterance_string: Combined string of all customer utterances
        agent_utterance_string: Combined string of all agent utterances
        customer_utterance_list: List of customer utterance dictionaries
        agent_utterance_list: List of agent utterance dictionaries
    Returns: Dictionary with matched_score, similarity_score, and overall_attention
    '''
    matched_score = keyword_score(customer_utterance_string, agent_utterance_string)
    
    sim_score = similarity_score(customer_utterance_list, agent_utterance_list)
    
    overall_attn = overall_attention(sim_score, matched_score)

    attention_dict = {
        'matched_score': matched_score,
        'similarity_score': sim_score,
        'overall_attention': overall_attn
    }
    return attention_dict



def Empathy(dialogue_diarized_string):
    '''
    Calculate empathy score from dialogue.

    Args: dialogue_diarized_string: String with CUSTOMER and AGENT labels

    Returns: Final empathy score)
    '''
    empathy_dict = empathy_check(dialogue_diarized_string=dialogue_diarized_string)

    emotion_recognition = float(empathy_dict.get('emotion_recognition', 0))
    emotion_validation = float(empathy_dict.get('emotion_validation', 0))
    support_intent = float(empathy_dict.get('support_intent', 0))

    final_empathy_score = emotion_recognition + emotion_validation + support_intent
    return final_empathy_score/3


def Greet_Ownership(agent_utterance_list):
    '''
    Calculate greeting and ownership scores.
    
    Args: agent_utterance_list: List of agent utterance dictionaries
    
    Returns: Tuple of (greet_score, ownership_score)
    '''
    greet_score = check_greetings(agent_utterance_list)
    ownership_score = check_ownership(agent_utterance_list)
    return greet_score, ownership_score


def Interuptions(corrected_utterances):
    """
    Check for interruptions in the conversation.
    
    Args:
        corrected_utterances: List of utterance dictionaries with speaker labels
    
    Returns:
        bool: True if interruption detected, False otherwise
    """
    interuption_bool = False  # Initialize to False (no interruption by default)
    
    for i, u in enumerate(corrected_utterances):
        if i+1 < len(corrected_utterances) and u.get('speaker') == 'Customer' and corrected_utterances[i+1]['speaker'] == 'Customer Service Agent':
            if corrected_utterances[i+1]['start'] - u.get('end') < 300:
                interuption_bool = True
                break  # Found interruption, no need to continue
                
    return interuption_bool

def Satisfaction(customer_utterance_list, portion=0.3):
    """
    Calculate customer satisfaction score.
    
    Args:
        customer_utterance_list: List of customer utterance dictionaries
        portion: Portion of conversation to analyze (default 0.3 = last 30%)
    
    Returns:
        Final satisfaction score (0-1)
    """
    explicit_score = explicit_check(customer_utterance_list, portion=portion)
    implicit_score = implicit_check(customer_utterance_list, portion=portion)
    final_satisfaction_score = (explicit_score + implicit_score) / 2
    return final_satisfaction_score


def Talk_to_listen_ratio(dialogue_string, dialogue_dict):
    """
    Calculate talk-to-listen ratio score.
    
    Args:
        dialogue_string: String representation of dialogue
        dialogue_dict: Full dialogue dictionary with utterances
    
    Returns:
        0 if unhealthy ratio, 1 if healthy ratio (0.3-0.7)
    """
    t2l = talk_to_listen(dialogue_string, dialogue_dict)
    if t2l > 0.7 or t2l < 0.3:
        return 0
    else:
        return 1 