from Transcript_actions.Speaker_classification import find_speaker, corrected_list

def interuptions(corrected_utterances):
    interuption_bool=False
    for i,u in enumerate(corrected_utterances):
        if i+1 < len(corrected_utterances) and u.get('speaker')=='Customer' and corrected_utterances[i+1]['speaker']=='Customer Service Agent':
            if corrected_utterances[i+1]['start']-u.get('end')<150:
                interuption_bool=True
                
    return interuption_bool