import os
import json
from dotenv import load_dotenv
import logging

 
from transcription_pipeline import AudioTranscription
from Transcript_actions.Speaker_classification import (
    find_speaker, 
    String_4_Semantic_analysis, 
    corrected_list, 
    customer_list_dict, 
    agent_list_dict
)
from evaluate import (
    Normalize_attention, 
    Empathy, 
    Greet_Ownership, 
    Interuptions,
    Satisfaction, 
    Talk_to_listen_ratio
)

def load_api_key():
    load_dotenv('key.env')
    api_key=os.getenv('ASSEMBLY_AI_KEY')
    return api_key

logging.basicConfig(
    level=logging.DEBUG
)
logger=logging.getLogger('uvicorn')

def Metrics(API_key:str, temp_path1:str):
    '''
    Transcription -> Diarization -> Metrics evaluation
    '''
    try:
        transcription=AudioTranscription(api_key=API_key)
    except Exception as e:
        logger.exception('Failed to initiate transcription %s', type(e).__name__)
     
    try:
        upload_url = transcription.upload_audio(audio_path=temp_path1)
    except Exception as e:
        logger.exception('Failed to fetch thje upload url from Assembly AI %s', type(e).__name__)


    transcription_id=transcription.perform_transcription(upload_url=upload_url)
    transcript_dict=transcription.get_transcript(transcription_id=transcription_id)

    undiarized_dialogue_string=transcription.string_4_speaker_Classification(transcription_process=transcript_dict)
    
    diarization_result=find_speaker(dialogue_string=undiarized_dialogue_string)

    diarized_dialogue_string=String_4_Semantic_analysis(dialogue_dict=transcript_dict, output=diarization_result)

    diarized_utterance_list=corrected_list(dialogue_dict=transcript_dict, output=diarization_result)
    customer_utterance_list, customer_utterance_string=customer_list_dict(corrected_list=diarized_utterance_list)
    agent_utterance_list, agent_utterance_string=agent_list_dict(corrected_list=diarized_utterance_list)
    
    # attention_dict = {
    #     'matched_score': matched_score,
    #     'similarity_score': sim_score,
    #     'overall_attention': overall_attn
    # }
    Attention_dict=Normalize_attention(customer_utterance_string, agent_utterance_string, customer_utterance_list, agent_utterance_list)
    overall_attention_score=Attention_dict.get('overall_attention')

    Empathy_score=Empathy(dialogue_diarized_string=diarized_dialogue_string)

    greet_score, ownership_score=Greet_Ownership(agent_utterance_list=agent_utterance_list)

    interuption_score=Interuptions(corrected_utterances=diarized_utterance_list)
    
    satisfaction_score=Satisfaction(customer_utterance_list=customer_utterance_list, portion=0.35)

    Talk_to_listen= Talk_to_listen_ratio(agent_utterance_list=agent_utterance_list, customer_utterance_list=customer_utterance_list)

    Evaluation_dict={
        'attention score' : overall_attention_score,
        'empathy score' : Empathy_score,
        'greet score' : greet_score,
        'ownership score' : ownership_score,
        'interuption score' : interuption_score,
        'satisfaction score' : satisfaction_score,
        'Talk to Listen' : Talk_to_listen  
    }
    return Evaluation_dict.json()
