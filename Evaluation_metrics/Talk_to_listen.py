def talk_to_listen(agent_utterance_list, customer_utterance_list)-> int:
    """
    > 0.7 → Customer dominates → agent may not be guiding to resolution

    0.3-0.7 → Healthy dialogue

    < 0.3 → Agent dominating → potential over-talking
    """
    agent_word_count=0    
    for a in agent_utterance_list:
        agent_text=a.get('text')
        agent_words=agent_text.split(' ')
        agent_word_countword_count+=len(agent_words)
    
    customer_word_count=0
    for c in customer_utterance_list:
        customer_text=c.get('text')
        customer_words=customer_text.split(' ')
        customer_word_count+=len(customer_words


