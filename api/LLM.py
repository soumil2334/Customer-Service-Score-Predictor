import json
import requests

def Summarize(metrics_strings_dictionary):

    Attention_string=metrics_strings_dictionary['Attention_string']
    Empathy_string=metrics_strings_dictionary['Empathy_string']
    Greeting_string=metrics_strings_dictionary['Greeting_string']
    Ownership_string=metrics_strings_dictionary['Ownership_string']
    Interruption_string=metrics_strings_dictionary['Interruption_string']
    Satisfaction_string=metrics_strings_dictionary['Satisfaction_string']
    Listening_ratio=metrics_strings_dictionary['Listening_ratio']

    prompt=f'''
ROLE:
You are an impartial performance evaluation assistant.
Your task is to summarize a customer care agent’s performance strictly based on the provided metrics.
Do NOT infer information that is not explicitly stated.
Do NOT introduce new examples or assumptions.
Maintain a professional, analytical tone.

You are given multiple evaluation metrics of a customer care agent derived from a real conversation.
Each metric includes either a score, supporting evidence, or both.

Metrics Provided:
1. Attention Evaluation
2. Empathy Evaluation
3. Greeting Behavior
4. Ownership Demonstration
5. Customer Interruptions
6. Customer Satisfaction
7. Talk-to-Listen Ratio

Below are the metric-wise observations:

ATTENTION:
{Attention_string}

EMPATHY:
{Empathy_string}

GREETING:
{Greeting_string}

OWNERSHIP:
{Ownership_string}

INTERRUPTIONS:
{Interruption_string}

CUSTOMER SATISFACTION:
{Satisfaction_string}

TALK-TO-LISTEN RATIO:
Listening ratio score: {Listening_ratio}

Instructions:
- Summarize the agent’s overall performance in **one cohesive paragraph**.
- Explicitly reference **strengths** and **areas of improvement**.
- If a metric is absent or negative, state it clearly without softening.
- Do not restate raw percentages unless necessary for clarity.
- Avoid bullet points; write in evaluative prose.

Return ONLY the final performance summary.
'''
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
        response_dict=json.loads(line)
        output+=str(response_dict['response'])

    return output
