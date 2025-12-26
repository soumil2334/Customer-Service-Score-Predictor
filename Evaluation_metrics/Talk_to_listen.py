from Transcript_actions.Speaker_classification import find_speaker

def talk_to_listen (dialogue_string:str, dialogue_dict:dict)-> int:
    """
    > 0.7 → Customer dominates → agent may not be guiding to resolution

    0.3-0.7 → Healthy dialogue

    < 0.3 → Agent dominating → potential over-talking
    """    
    utterances_list=dialogue_dict.get('utterances')

    classification_dict=find_speaker(dialogue_string)
    A=classification_dict['Speaker A']
    B=classification_dict['Speaker B']

    A_count=0
    B_count=0

    for u in utterances_list:
        if u.get('speaker')=='A':
            A_count+=len(u.get('text'))
        if u.get('speaker')=='B':
            B_count+=len(u.get('text'))
    
    if A=='Customer':
        Customer_count=A_count
        Agent_count=B_count
    elif B=='Customer':
        Customer_count=B_count
        Agent_count=A_count
    else:
        Customer_count=0
        Agent_count=0

    return Customer_count/(1+Agent_count)



