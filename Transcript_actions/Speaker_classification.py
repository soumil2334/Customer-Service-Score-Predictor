import requests
import json

def find_speaker(dialogue_string:str) :
    prompt=f"""
    ROLE:
    You are a specialist in analysing cutomer care calls, therefore you will be provided by a transcript string and you have to
    identify between the two speakers who is the Customer Agent and who is the customer.

    CONTEXT:
    The string will be a transcript with two Speakers A and B, analyse the way they speak and what they speak to conclude who is who.
    If a speaker is putting forward his/her complains or is asking for some help/query then that speaker has a high probability to be 
    the customer but if he/she is telling some solutions or is guiding the other speaker, then he/she could be the Customer agent.

    CONSTRAINTS: 
    - if not sure then take the decision with the most likely possiblity
    - take the final decision after analysing all the lines of the conversation carefully
    - output should be in the form of JSON:
    OUTPUT FORMAT - 
    { 'Speaker A' : 'Customer', 'Speaker B' : 'Customer Service Agent', 'Confidence': 'for example 92%'}

    INPUT:
    Transcipt : {dialogue_string}
    """

    Ollama_response=requests.post(
        url= 'http://localhost:11434/api/generate',
        json={
            'model': 'llama3',
            'prompt': prompt,
            'temperature': 0
        }
    )

    #Ollama returns out put line by line in this form:

# {"response":"{","done":false}
# {"response":"\"customer\":","done":false}
# {"response":"\"Speaker B\"","done":false}
# {"response":",","done":false}
# {"response":"\"agent\":","done":false}
# {"response":"\"Speaker A\"","done":false}
# {"response":"}","done":false}
# {"done":true}
    output=''
    for line in Ollama_response.iter_lines():
        if line:
             # we need to do this because sometime the HTTP returns empty strings that might lead to json CRASH when we do json.loads()
            data=json.loads(line) # converts the line(bytes here in json format) into dictionary
            output+=str(data['response'])
    
    return json.loads(output)

def String_4_Semantic_analysis(dialogue_dict:dict, output:dict):
    string1=''
    A=output.get('Speaker A')
    B=output.get('Speaker B')
    utterances_list=dialogue_dict.get('utterances')
    for u in utterances_list:
        if u.get('speaker')=='A': 
            string1+=f"{A}: {u.get('text')}\n"

        elif u.get('speaker')=='B': 
            string1+=f"{B}: {u.get('text')}\n"
    
    return string1

def corrected_list(dialogue_dict:dict, output:dict):
    utterances_list=dialogue_dict.get('utterances')
    for u in utterances_list:
        if u.get('speaker')=='A':
            u['speaker']=output.get('Speaker A')
        if u.get('speaker')=='B':
            u['speaker']=output.get('Speaker B')

    return utterances_list


def customer_list_dict(corrected_list):
    text=''
    list_2=[]
    for c in corrected_list:
        if c.get('speaker')=='Customer':
            list_2.append(c)
            text=text+str(c.get('text'))+'\n'
    return list_2, text

def agent_list_dict(corrected_list):
    text=''
    list_2=[]
    for c in corrected_list:
        if c.get('speaker')=='Customer Service Agent':
            list_2.append(c)
            text=text+str(c.get('text'))+'\n'
    return list_2, text
