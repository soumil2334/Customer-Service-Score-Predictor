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
        start=c.get('start')
        end=c.get('end')
        total=end-start

        customer_time+=total

    total_time=customer_time+agent_time

    if total_time <= 0:
    # no speech, return neutral score
        return 0.5
    
    ratio=customer_time/total_time

    #triangular mapping -- ratio=0.5 ideal case score=1
    #                   -- ratio=1.0 customer the only one speaking score=0
    #                   -- ratio=0 agent dominating score=0
    score = max(0.0, 1.0 - 2.0 * abs(ratio - 0.5))

    return round(score, 2)