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
from Evaluation_metrics.Main_evaluation import (
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
        logger.info("Initiating transcription")
        transcription=AudioTranscription(api_key=API_key)
        upload_url = transcription.upload_audio(audio_path=temp_path1)
        logger.info(f'Upload URL : {upload_url}')   
    
        logger.info("Fetching transcription ID from Assembly AI")
        transcription_id=transcription.perform_transcription(upload_url=upload_url)
        transcript_dict=transcription.get_transcript(transcription_id=transcription_id)

        logger.info("Diarization via LLM")
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

        logger.info('Calculating the various metrics')
        Attention_dict=Normalize_attention(customer_utterance_string, agent_utterance_string, customer_utterance_list, agent_utterance_list)
        overall_attention_score=Attention_dict.get('overall_attention')
        Empathy_score=Empathy(dialogue_diarized_string=diarized_dialogue_string)
        greet_score, ownership_score=Greet_Ownership(agent_utterance_list=agent_utterance_list)
        interuption_score=Interuptions(corrected_utterances=diarized_utterance_list)
        satisfaction_score=Satisfaction(customer_utterance_list=customer_utterance_list, portion=0.35)
        Talk_to_listen= Talk_to_listen_ratio(agent_utterance_list=agent_utterance_list, customer_utterance_list=customer_utterance_list)

        Evaluation_dict = {
            'attention score': overall_attention_score,
            'empathy score': Empathy_score,
            'greet score': greet_score,
            'ownership score': ownership_score,
            'interuption score': interuption_score,
            'satisfaction score': satisfaction_score,
            'Talk to Listen': Talk_to_listen
        }
        
        # Validation to mke sure all values are in b/w [0,1] 
        for metric_name, score in Evaluation_dict.items():
            if not isinstance(score, (int, float)):
                logger.warning(f"{metric_name} is not a number: {score} (type: {type(score)})")
            elif score < 0 or score > 1:
                logger.warning(f"{metric_name} is outside [0,1] range: {score}")
        
        return Evaluation_dict

    except Exception as e:
        logger.exception(f'Exception {type(e).__name__} has occurred')
        raise

def Final_score(Evaluation_dict:dict):
    #randomnly assigned weights to the various score
    weights={
        'attention_score' : 0.2,
        'empathy_score' : 0.2,
        'greet_score' : 0.1,
        'ownership_score' : 0.15,
        'interuption_score' : 0.1,
        'satisfaction_score' : 0.15,
        'Talk To Listen' : 0.1
    }

    attention_score=Evaluation_dict['attention score']*weights['attention_score']
    empathy_score=Evaluation_dict['empathy score']*weights['empathy score']
    greet_score=Evaluation_dict['greet score']*weights['greet score']
    ownership_score=Evaluation_dict['ownership score']*weights['ownership score']
    interuption_score=Evaluation_dict['interuption score']*weights['interuption score']
    satisfaction_score=Evaluation_dict['satisfaction score']*weights['satisfaction score']
    Listening_score=Evaluation_dict['Talk to Listen']*weights['Talk to Listen']

    final_score=attention_score + empathy_score + greet_score + ownership_score + interuption_score + satisfaction_score + Listening_score

    final_output={
        'Final Agent Score' : final_score,
        'Breakdown' : {
            'Agent Attention Score' : attention_score,
            'Agent Empathy Score' : empathy_score,
            'Interuption by Agent' : interuption_score,
            'Satisfaction of the Customer' : satisfaction_score,
            'Agent Listening Score ': Listening_score,
            'Did the Agent greet' : bool(Evaluation_dict['greet score']),
            'Did the Agent took Ownership' : bool(Evaluation_dict['ownership score'])
        },
        'Individual Score': Evaluation_dict
    }

    return final_output