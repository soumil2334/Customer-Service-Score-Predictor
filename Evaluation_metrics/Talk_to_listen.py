def talk_to_listen(agent_utterance_list, customer_utterance_list)-> float:
    """
    > 0.7 → Customer dominates → agent may not be guiding to resolution

    0.3-0.7 → Healthy dialogue

    < 0.3 → Agent dominating → potential over-talking
    """
    agent_time=0
    customer_time=0

    for a in agent_utterance_list:
        start=a.get('start')
        end=a.get('end')
        total=end-start

        agent_time+=total
    
    for c in customer_utterance_list:
        start=a.get('start')
        end=a.get('end')
        total=end-start

        customer_time+=total
    
    ratio=customer_time/agent_time

    return round(ratio, 2)

