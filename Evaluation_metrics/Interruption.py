def interuptions(corrected_utterances, tolerance):
    customer_turns=0
    interuption_count=0
    interuption_sentences=[]
    interuption_time=[]
    for i,u in enumerate(corrected_utterances):
        if i+1<len(corrected_utterances):
            speaker_1=u.get('speaker')
            speaker_2=corrected_utterances[i+1].get('speaker')

            if speaker_1=='Customer' and speaker_2=='Customer Service Agent':
                customer_turns+=1

                if corrected_utterances[i].get('end') - 100 > corrected_utterances[i+1].get('start'):
                    
                    interuption_sentences.append(str(f'''{corrected_utterances[i].get('text')}\n{corrected_utterances[i+1].get('text')}'''))
                    
                    interuption_time.append(corrected_utterances[i+1].get('start'))
                    
                    interuption_count+=1
        else:
            break

    if interuption_count>=1:
        interuption_score=0
    else:
        interuption_score=1
    return interuption_score, interuption_sentences