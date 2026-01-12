import os
import logging
import io
import json
from dotenv import load_dotenv
import logging
from pydantic import BaseModel, Field
from Transcript_actions.transcription_pipeline import AudioTranscription
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

logging.basicConfig(
    level=logging.DEBUG
)
logger=logging.getLogger('uvicorn')

class Score_result(BaseModel):
    Attention : str=Field(alias='Attention_string')
    Empathy : str=Field(alias='Empathy_string')
    Greeting : str=Field(alias='Greeting_string')
    Ownership : str=Field(alias='Ownership_string')
    Interruption : str=Field(alias='Interruption_string')
    Satisfaction : str=Field(alias='Satisfaction_string')
    Listening : str=Field(alias='Listening_ratio')

def load_api_key():
    load_dotenv('key.env')
    api_key=os.getenv('ASSEMBLY_AI_KEY')
    return api_key

def matplot_figure_2_png(matplot_figure):
    buf=io.BytesIO()
    matplot_figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    return buf

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
        #     'overall_attention': overall_attn}

        logger.info('Calculating the various metrics')
        
        Attention_dict, attention_sentences=Normalize_attention(customer_utterance_string, agent_utterance_string, customer_utterance_list, agent_utterance_list)
        overall_attention_score=Attention_dict.get('overall_attention')
        
        Empathy_score, empathy_sentences=Empathy(dialogue_diarized_string=diarized_dialogue_string)
        
        greet_score_sentence, ownership_score_sentence=Greet_Ownership(agent_utterance_list=agent_utterance_list)
        
        interuption_count, interuption_sentence_dict=Interuptions(corrected_utterances=diarized_utterance_list)
        
        satisfaction=Satisfaction(customer_utterance_list=customer_utterance_list, portion=0.35)
        
        Talk_to_listen= Talk_to_listen_ratio(agent_utterance_list=agent_utterance_list, customer_utterance_list=customer_utterance_list)
         
        # Sentence proofs to be shown at the frontend
        attention_display=max(attention_sentences, key= lambda x: x[0])
        empathy_display=empathy_sentences
        greet_sentence=greet_score_sentence[1]
        ownership_sentence=ownership_score_sentence[1]
        interuption=interuption_sentence_dict

        Evaluation_dict = {
            'attention_score': overall_attention_score,
            'empathy_score': Empathy_score,
            'greet_score': greet_score_sentence[0],
            'ownership_score': ownership_score_sentence[0],
            'interuption_count': interuption_count,
            'satisfaction_score': satisfaction[0],
            'Talk_to_Listen': Talk_to_listen
        }

        extra_metric={
            'attention' : attention_display,
            'empathy' : empathy_display,
            'interuption' : interuption_sentence_dict,
            'Greet': greet_sentence,
            'Ownership' : ownership_sentence,
            'interruption' : interuption,
            'satisfaction_graph': satisfaction[1]
 }
        
        # Validation to mke sure all values are in b/w [0,1] 
        for metric_name, score in Evaluation_dict.items():
            if not isinstance(score, (int, float)):
                logger.warning(f"{metric_name} is not a number: {score} (type: {type(score)})")
            elif score < 0 or score > 1:
                logger.warning(f"{metric_name} is outside [0,1] range: {score}")
        
        return Evaluation_dict, extra_metric

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


def metrics_calculation(file_path):
    try:
        api_key=load_api_key()
        evaluation_dict, extra_dict=Metrics(api_key, file_path)
        
        #Attention
        attention_score= evaluation_dict['attention score']
        attention_sentence_display= extra_dict['attention']
        attention_message=str(f'''Overall attention score is {attention_score*100}%\n
                                For instance : Agent displayed a score of 
                              {attention_sentence_display[0]*100}% during this 
                              part of the conversation :- \n {attention_sentence_display[1]}''')    

        #Empathy   
        Empathy_score=evaluation_dict['empathy score']
        empathy_sentence_display= extra_dict['empathy']
        empathy_message=str(f'''The overall empathy score is {Empathy_score * 100}% for instance : \n {str(empathy_sentence_display)}''')

        #Greeting
        greeting_score=evaluation_dict['greet score']
        greet_sentence_display=extra_dict['Greet']
        if greeting_score:
            greet_message=str(f'''Did the agent greet the customer : {bool(greeting_score)}\n {greet_sentence_display}''')
        else:
            greet_message=str(f'''Did the agent greet the customer : {bool(greeting_score)}''')
    
        #Ownership
        ownership_score=evaluation_dict['ownership score']
        ownership_sentence_display=extra_dict['Onwership']
        if ownership_score:
            ownership_message=str(f'''The overall agent's ownership score is {ownership_score}\n {ownership_sentence_display}''') 
        else: 
            ownership_message=str(f'''The agent didn't take ownership''')           
    
        #Interuption
        interruption_count=evaluation_dict['interuption count']
        interruption_sentences_display=extra_dict['interruption']
            
        if interruption_count:
            output_interruption= '\n\n'.join(interruption_sentences_display)
            interruption_message=str(f''' The Agent interrupted the customer {interruption_count} time{'s' if interruption_count > 1 else ''} at {output_interruption} ''')
        else: 
            interruption_message=str(f'''The agent didn't interrupt the customer''')
        
        #Satisfaction
        satisfaction_score=evaluation_dict['satisfaction score']
        satisfaction_graph_figure=extra_dict['satisfaction_graph']
        satisfaction_message=str(f'''The satisfaction score of the customer was {satisfaction_score}''')
        
        #Talk_to_listen_ratio
        talk_to_listen_score=evaluation_dict['Talk to Listen']
   
        response=Score_result(
            Attention_string = attention_message,
            Empathy_string = empathy_message,
            Greeting_string = greet_message,
            Ownership_string = ownership_message,
            Interruption_string = interruption_message,
            Satisfaction_string = satisfaction_message,
            Listening_ratio = talk_to_listen_score
        )
        
        satisfaction_graph_stream=matplot_figure_2_png(satisfaction_graph_figure)

        return response, satisfaction_graph_stream
    
    except Exception as e:
        logging.exception(f'''Due to {type(e).__name__} metrics calculation couldn't be carried forward''')
        raise