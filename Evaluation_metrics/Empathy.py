import requests
import ast
import json
import numpy as np
from pydantic import BaseModel, TypeAdapter, Field, ValidationError
from typing import List
import logging

logging.basicConfig(level=logging.DEBUG, format=(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
))
logger=logging.getLogger(__name__)

class Empathy_Metrics(BaseModel):
    emotion_recognition : float= Field(alias='emotion_recognition')
    emotion_validation : float= Field(alias='emotion_validation')
    support_intent : float= Field(alias='support_intent')
    final_empathy_score : float= Field(alias='final_empathy_score')
    Valid_Reason : str= Field(alias='Valid Reason')

class LLM_response(BaseModel):
    Customer_message : str=Field(alias='Customer message')
    Agent_reponse : str=Field(alias='Agent response')
    Empathy : Empathy_Metrics=Field(alias='Empathy')

adapter=TypeAdapter(List[LLM_response])

def empathy_check(dialogue_diarized_string):

    prompt=f"""
You are an impartial quality auditor evaluating empathy in a customer service call.

You will receive a full transcript containing CUSTOMER and CUSTOMER SERVICE AGENT turns.

Instructions:
1. Read the transcript turn-by-turn.
2. For each AGENT response, evaluate empathy toward the previous CUSTOMER message.
3. Empathy has three dimensions:
   - Emotional Recognition
   - Emotional Validation
   - Supportive Intent
4. Assign empathy score to all the individual agent response to the customer's message.
5. Score EACH dimension between 0 and 1.
   - Assign 1 ONLY if supported by the agent's exact words.
   - If no evidence exists, score must be 0.
   - Politeness alone is NOT empathy.
6. Calculate the final agent empathy score as the average across all agent turns.
7. Provide valid reason/justification for assigning the score. 

Return a list of all Agent responses to the customer messages as per the provided Transcript along with the
Empathy JSON in the following format:

[{'Customer message': str(),
  'Agent response' : str(),
  'Empathy':{
    "emotion_recognition": 0-1,
    "emotion_validation": 0-1,
    "support_intent": 0-1,
    "final_empathy_score": 0-1,
    "Valid Reason": str()}
},
{ 'Customer message': str(),
  'Agent response' : str(),
  'Empathy':{
    "emotion_recognition": 0-1,
    "emotion_validation": 0-1,
    "support_intent": 0-1,
    "final_empathy_score": 0-1,
    "Valid Reason": str()}
}
.....
]

Transcript:
{dialogue_diarized_string}
"""
    response=requests.post(
        url='http://localhost:11434/api/generate',
        json={
            'model':'llama3',
            'prompt': prompt,
            'temperature': 0   
        },
        stream=True
    )

    # Here i have iterated over the line becasue llama APIs return the output in chunks 
    output=''
    for line in response.iter_lines():
        if line:
            data=json.loads(line)
            output+=str(data['response'])      
    output_list=ast.literal_eval(output)
    
    score=[]
    try:
        validate_output=adapter.validate_python(output_list)
        for dictionary in validate_output:
            Empathy=dictionary.get('Empathy')
            Valid_reason=Empathy.get('Valid Reason')
            if Valid_reason:
                score.append(float(Empathy.get('final_empathy_score')))
    
    except ValidationError as e:
        logger.error(f'Unexpected error {e} occurred')
        raise
    
    score_value=np.mean(score)
    return score_value

        


