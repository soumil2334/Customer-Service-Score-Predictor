import math
from torch.nn import attention
from transcription_pipeline import AudioTranscription
from Evaluation_metrics.Attention import keyword_extractor, keyword_score, similarity_score, overall_attention
from Evaluation_metrics.Empathy import empathy_check
from Evaluation_metrics.Greetings_ownership import check_greetings, check_ownership
from Evaluation_metrics.Interruption import interuptions
from Evaluation_metrics.satisfaction import keywords_func, sentiment_score, explicit_check, implicit_check
from Evaluation_metrics.Talk_to_listen import talk_to_listen

def Normalize_attention(customer_utterance_string, agent_utterance_string, customer_utterance_ist, agent_utterance_list):
    matched_score=keyword_score(customer_utterance_string, agent_utterance_string)
    
    similarity_score=similarity_score(customer_utterance_list, agent_utterance_list)
    
    overall_attention=overall_attention(similarity_score, matched_score)

    attention_dict={
        'matched_score': matched_score,
        'similarity_score': similarity_score,
        'overall_attention': overall_attention
    }
    return attention_dict


def Empathy(dialogue_diarized_string):
    empathy_dict=empathy_check(dialogue_string)

    emotion_recognition=int(empathy_dict.get('emotion_recognition'))
    emotion_validation=int(empathy_dict.get('emotion_validation'))
    support_intent=int(empathy_dict.get('support_intent'))

    final_empathy_score= emotion_recognition+emotion_validation+support_intent
    return final_empathy_score


def Greet_Ownership(agent_utterance_list):
    greet_score=check_greetings(agent_list)

    ownership_score=check_ownership(agent_list)
    return greet_score, ownership_score


def Interuptions(dialogue_dict, dialogue_string):
    interuption_bool=interuptions(dialogue_dict, dialogue_string)
    
    if interuption_bool==False:
        return 0
    elif interuption_bool==True:
        return 1 


def Satisfaction(customer_utterance_list, portion=0.3):
    explicit_check=explicit_check(customer_dict_list, portion=0.35)
    implicit_check=implicit_check(customer_dict_list, portion=0.35)
    final_satisfaction_score=(explicit_check+implicit_check)/2
    return final_satisfaction_score

def Talk_to_listen(dialogue_string, dialogue_dict):
    t2l=talk_to_listen(dialogue_string, dialogue_dict)
    if t2l>0.7 OR t2l<0.3:
        return 0
    else:
        return 1 