import math
from Evaluation_metrics.Attention import keyword_score, similarity_score, overall_attention
from Evaluation_metrics.Empathy import empathy_check
from Evaluation_metrics.Greetings_ownership import check_greetings, check_ownership
from Evaluation_metrics.Interruption import interuptions
from Evaluation_metrics.satisfaction import sentiment_trajectory, explicit_check, implicit_check
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
    sim_score, sentences = similarity_score(customer_utterance_list, agent_utterance_list)

    overall_attn = overall_attention(sim_score, matched_score)

    attention_dict = {
        'matched_score': matched_score,
        'similarity_score': sim_score,
        'overall_attention': overall_attn
    }
    return attention_dict, sentences



def Empathy(dialogue_diarized_string):
    '''
    Calculate empathy score from dialogue.

    Args: dialogue_diarized_string: String with CUSTOMER and AGENT labels

    Returns: Final empathy score)
    '''
    empathy_dict = empathy_check(dialogue_diarized_string=dialogue_diarized_string)
     
    emotion_recognition, max_empathy_convo = float(empathy_dict.get('emotion_recognition', 0))
    emotion_validation = float(empathy_dict.get('emotion_validation', 0))
    support_intent = float(empathy_dict.get('support_intent', 0))
    
    final_empathy_score = emotion_recognition + emotion_validation + support_intent
    return final_empathy_score/3, max_empathy_convo


def Greet_Ownership(agent_utterance_list):
    '''
    Calculate greeting and ownership scores.
    
    Args: agent_utterance_list: List of agent utterance dictionaries
    
    Returns: Tuple of (greet_score, ownership_score)
    '''
    greet_score_sentence = check_greetings(agent_utterance_list)
    # both functions return score on index 0 and sentence at index 1
    ownership_score_sentence = check_ownership(agent_utterance_list)
    return greet_score_sentence, ownership_score_sentence


def Interuptions(corrected_utterances, tolerance):
    '''Interuption_score represents the number of time the speaker was interupted 
    and the interuption_time represenets hte time when the agent was interupted'''
    interuption_count, interuption_sentence_dict=interuptions(corrected_utterances, tolerance)
    return interuption_count, interuption_sentence_dict

def Satisfaction(customer_utterance_list, portion=0.3):
    """
    Calculate customer satisfaction score and show the emotion trajectory
    A trajectory thatis gradually moving upwards in +ve possibly symbolises growing satisfaction
    Args:
        customer_utterance_list: List of customer utterance dictionaries
        portion: Portion of conversation to analyze (default 0.3 = last 30%)
    
    Returns:
        Final satisfaction score (0-1), Satisfaction trajecory of the customer
    """

    trajectory = sentiment_trajectory(customer_utterance_list)

    explicit_score = explicit_check(customer_utterance_list, portion=portion)
    implicit_score = implicit_check(customer_utterance_list, portion=portion)
    final_satisfaction_score = (explicit_score + implicit_score) / 2

    return final_satisfaction_score, trajectory


def Talk_to_listen_ratio(agent_utterance_list, customer_utterance_list):
    '''
    Customer dominates (> 0.7 customer share)+
    What it usually means--
       Customer is narrating, venting, or repeating
       Agent is mostly listening or acknowledging
       Problem may not be structured yet

    Balanced (≈ 0.3 – 0.7)
    What it usually means--
       Customer explains
       Agent probes, clarifies, and guides
       Information exchange is bidirectional

    Agent dominates (< ~0.3 customer share)
    What it usually means--
       Agent is over-explaining or scripting
       Customer not given space to clarify
       High risk of misunderstanding
    '''

    # ratio is customer_speaking_time/ agent_speaking_time
    score=talk_to_listen(agent_utterance_list, customer_utterance_list)

    #while returning the final ratio also return the final conclusion 
    #the ratio alone is useless without the explanation for the user
    #try using an LLM for better undewrstanding

    return score