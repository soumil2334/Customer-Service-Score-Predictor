import requests
import json

def empathy_check(dialogue_diarized_string):

    prompt=f"""
You are an impartial quality auditor evaluating empathy in a customer service call.

You will receive a full transcript containing CUSTOMER and AGENT turns.

Instructions:
1. Read the transcript turn-by-turn.
2. For each AGENT response, evaluate empathy toward the immediately preceding CUSTOMER message.
3. Empathy has three dimensions:
   - Emotional Recognition
   - Emotional Validation
   - Supportive Intent
4. Score EACH dimension as either 0 or 1.
   - Assign 1 ONLY if supported by the agent's exact words.
   - If no evidence exists, score must be 0.
   - Politeness alone is NOT empathy.
5. Calculate the final agent empathy score as the average across all agent turns.

Return JSON only in the following format:

{
    "emotion_recognition": 0-1,
    "emotion_validation": 0-1,
    "support_intent": 0-1,
    "final_empathy_score": 0-1

}

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
    output=''
    for line in response.iter_lines():
        if line:
            data=json.loads(line)
            output+=str(data['response'])
    
    return json.loads(output)
