import os
import json
from dotenv import load_dotenv
from transcription_pipeline import AudioTranscription
from Transcript_actions.Speaker_classification import (
    find_speaker, 
    String_4_Semantic_analysis, 
    corrected_list,
    customer_list_dict,
    agent_list_dict
)
from evaluate import (
    Normalize_attention,
    Empathy,
    Greet_Ownership,
    Interuptions,
    Satisfaction,
    Talk_to_listen_ratio
)


def load_api_key():
    """Load API key from environment file."""
    load_dotenv('key.env')
    api_key = os.getenv('ASSEMBLY_AI_KEY')
    if not api_key:
        raise ValueError("ASSEMBLY_AI_KEY not found in key.env file")
    return api_key


def transcribe_audio(audio_path: str, api_key: str, output_json_path: str = "transcript_output.json"):
    """
    Step 1: Transcribe audio file and get diarized transcript.
    
    Args:
        audio_path: Path to audio file
        api_key: AssemblyAI API key
        output_json_path: Path to save transcript JSON
    
    Returns:
        Dictionary containing transcript data
    """
    print("=" * 60)
    print("STEP 1: Transcribing Audio File")
    print("=" * 60)
    
    transcriber = AudioTranscription(api_key=api_key)
    
    print(f"Uploading audio file: {audio_path}")
    upload_url = transcriber.upload_audio(audio_path)
    print("Audio uploaded successfully")
    
    print("Starting transcription...")
    transcription_id = transcriber.perform_transcription(upload_url)
    print(f"Transcription started (ID: {transcription_id})")
    
    print("Waiting for transcription to complete...")
    transcript_json = transcriber.get_transcript(transcription_id, output_json_path)
    print(f"Transcription completed and saved to {output_json_path}")
    
    return transcript_json


def classify_speakers(transcript_json: dict):
    """
    Step 2: Classify speakers (Customer vs Agent).
    
    Args:
        transcript_json: Transcript dictionary from AssemblyAI
    
    Returns:
        Tuple of (speaker_classification_dict, dialogue_string, corrected_utterances)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Classifying Speakers")
    print("=" * 60)
    
    # Convert transcript to dialogue string
    transcriber = AudioTranscription(api_key="dummy")  # Just for the method
    dialogue_string = transcriber.string_4_speaker_Classification(transcript_json)
    
    # Classify speakers using Ollama
    print("Classifying speakers using AI...")
    speaker_classification = find_speaker(dialogue_string)
    print(f"✓ Speakers classified: {speaker_classification}")
    
    # Get corrected list with proper speaker labels
    corrected_utterances = corrected_list(transcript_json.copy(), speaker_classification)
    
    # Create dialogue string with proper labels for empathy analysis
    dialogue_diarized_string = String_4_Semantic_analysis(transcript_json, speaker_classification)
    
    return speaker_classification, dialogue_string, dialogue_diarized_string, corrected_utterances


def separate_utterances(corrected_utterances: list):
    """
    Step 3: Separate customer and agent utterances.
    
    Args:
        corrected_utterances: List of utterances with corrected speaker labels
    
    Returns:
        Tuple of (customer_list, customer_string, agent_list, agent_string)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Separating Customer and Agent Utterances")
    print("=" * 60)
    
    customer_list, customer_string = customer_list_dict(corrected_utterances)
    agent_list, agent_string = agent_list_dict(corrected_utterances)
    
    print(f"✓ Found {len(customer_list)} customer utterances")
    print(f"✓ Found {len(agent_list)} agent utterances")
    
    return customer_list, customer_string, agent_list, agent_string


def calculate_all_metrics(
    transcript_json: dict,
    dialogue_string: str,
    dialogue_diarized_string: str,
    customer_list: list,
    customer_string: str,
    agent_list: list,
    agent_string: str
):
    """
    Step 4: Calculate all evaluation metrics.
    
    Args:
        transcript_json: Full transcript dictionary
        dialogue_string: Original dialogue string
        dialogue_diarized_string: Dialogue string with CUSTOMER/AGENT labels
        customer_list: List of customer utterance dictionaries
        customer_string: Combined customer text
        agent_list: List of agent utterance dictionaries
        agent_string: Combined agent text
    
    Returns:
        Dictionary with all evaluation scores
    """
    print("\n" + "=" * 60)
    print("STEP 4: Calculating Evaluation Metrics")
    print("=" * 60)
    
    scores = {}
    
    # 1. Attention Score
    print("Calculating Attention Score...")
    attention_dict = Normalize_attention(
        customer_utterance_string=customer_string,
        agent_utterance_string=agent_string,
        customer_utterance_list=customer_list,
        agent_utterance_list=agent_list
    )
    scores['attention'] = attention_dict
    print(f"✓ Attention Score: {attention_dict['overall_attention']:.3f}")
    
    # 2. Empathy Score
    print("Calculating Empathy Score...")
    empathy_score = Empathy(dialogue_diarized_string)
    scores['empathy'] = empathy_score
    print(f"✓ Empathy Score: {empathy_score:.3f}")
    
    # 3. Greetings and Ownership
    print("Calculating Greetings and Ownership Scores...")
    greet_score, ownership_score = Greet_Ownership(agent_list)
    scores['greetings'] = greet_score
    scores['ownership'] = ownership_score
    print(f"✓ Greetings Score: {greet_score:.3f}")
    print(f"✓ Ownership Score: {ownership_score:.3f}")
    
    # 4. Interruptions
    print("Checking for Interruptions...")
    interruption_score = Interuptions(transcript_json, dialogue_string)
    scores['interruptions'] = interruption_score
    print(f"✓ Interruption Score: {interruption_score}")
    
    # 5. Satisfaction
    print("Calculating Satisfaction Score...")
    satisfaction_score = Satisfaction(customer_list, portion=0.3)
    scores['satisfaction'] = satisfaction_score
    print(f"✓ Satisfaction Score: {satisfaction_score:.3f}")
    
    # 6. Talk-to-Listen Ratio
    print("Calculating Talk-to-Listen Ratio...")
    talk_listen_score = Talk_to_listen_ratio(dialogue_string, transcript_json)
    scores['talk_to_listen'] = talk_listen_score
    print(f"✓ Talk-to-Listen Score: {talk_listen_score}")
    
    return scores


def calculate_final_score(scores: dict):
    """
    Step 5: Calculate final composite score.
    
    Args:
        scores: Dictionary with all individual scores
    
    Returns:
        Final composite score and breakdown
    """
    print("\n" + "=" * 60)
    print("STEP 5: Calculating Final Composite Score")
    print("=" * 60)
    
    # Normalize scores to 0-1 range and calculate weighted average
    attention_score = scores['attention']['overall_attention']  # Already 0-1
    empathy_score = scores['empathy'] / 3.0  # Normalize from 0-3 to 0-1
    greetings_score = scores['greetings']  # Already 0-1
    ownership_score = scores['ownership']  # Already 0-1
    interruption_score = 1 - scores['interruptions']  # Invert (0=good, 1=bad)
    satisfaction_score = scores['satisfaction']  # Already 0-1
    talk_listen_score = scores['talk_to_listen']  # Already 0-1
    
    # ========================================================================
    # METRIC WEIGHTS - IMPORTANT: Adjust based on your business priorities
    # ========================================================================
    # These weights determine how much each metric contributes to the final score.
    # 
    # Current weights (default/placeholder - needs validation):
    # - Empathy (20%): High weight - emotional connection is critical
    # - Satisfaction (20%): High weight - ultimate customer outcome
    # - Attention (15%): Medium weight - agent must listen to customer
    # - Ownership (15%): Medium weight - agent takes responsibility
    # - Greetings (10%): Lower weight - important but less critical
    # - Interruptions (10%): Lower weight - negative behavior indicator
    # - Talk-to-Listen (10%): Lower weight - communication balance
    #
    # TO DETERMINE PROPER WEIGHTS:
    # 1. Business priorities: What matters most for your use case?
    # 2. Research/benchmarks: Use industry standards or academic research
    # 3. Data analysis: Correlate metrics with actual customer outcomes
    # 4. Expert consultation: Get input from customer service experts
    # 5. A/B testing: Test different weight combinations and measure results
    #
    # NOTE: Weights must sum to 1.0 (100%)
    # ========================================================================
    weights = {
        'attention': 0.15,      # Medium priority: Agent listens to customer
        'empathy': 0.20,        # High priority: Emotional connection
        'greetings': 0.10,      # Lower priority: Professional courtesy
        'ownership': 0.15,      # Medium priority: Taking responsibility
        'interruptions': 0.10,  # Lower priority: Negative behavior
        'satisfaction': 0.20,   # High priority: Customer outcome
        'talk_to_listen': 0.10  # Lower priority: Communication balance
    }
    
    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:  # Allow small floating point errors
        raise ValueError(f"Weights must sum to 1.0, but sum to {total_weight}")
    
    final_score = (
        attention_score * weights['attention'] +
        empathy_score * weights['empathy'] +
        greetings_score * weights['greetings'] +
        ownership_score * weights['ownership'] +
        interruption_score * weights['interruptions'] +
        satisfaction_score * weights['satisfaction'] +
        talk_listen_score * weights['talk_to_listen']
    )
    
    breakdown = {
        'attention': attention_score,
        'empathy': empathy_score,
        'greetings': greetings_score,
        'ownership': ownership_score,
        'interruptions': interruption_score,
        'satisfaction': satisfaction_score,
        'talk_to_listen': talk_listen_score
    }
    
    print(f"\n{'Metric':<20} {'Score':<10} {'Weight':<10}")
    print("-" * 40)
    for metric, score in breakdown.items():
        print(f"{metric.capitalize():<20} {score:<10.3f} {weights[metric]:<10.2f}")
    print("-" * 40)
    print(f"{'FINAL SCORE':<20} {final_score:<10.3f}")
    print("=" * 60)
    
    return final_score, breakdown


def save_results(scores: dict, final_score: float, breakdown: dict, output_path: str = "evaluation_results.json"):
    """
    Save evaluation results to JSON file.
    
    Args:
        scores: Dictionary with all individual scores
        final_score: Final composite score
        breakdown: Score breakdown
        output_path: Path to save results
    """
    results = {
        'final_score': final_score,
        'score_breakdown': breakdown,
        'detailed_scores': scores
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


def main(audio_path: str):
    """
    Main function to run the complete evaluation pipeline.
    
    Args:
        audio_path: Path to the audio file to evaluate
    """
    try:
        # Load API key
        api_key = load_api_key()
        
        # Step 1: Transcribe audio
        transcript_json = transcribe_audio(audio_path, api_key)
        
        # Step 2: Classify speakers
        speaker_classification, dialogue_string, dialogue_diarized_string, corrected_utterances = classify_speakers(transcript_json)
        
        # Step 3: Separate utterances
        customer_list, customer_string, agent_list, agent_string = separate_utterances(corrected_utterances)
        
        # Step 4: Calculate all metrics
        scores = calculate_all_metrics(
            transcript_json=transcript_json,
            dialogue_string=dialogue_string,
            dialogue_diarized_string=dialogue_diarized_string,
            customer_list=customer_list,
            customer_string=customer_string,
            agent_list=agent_list,
            agent_string=agent_string
        )
        
        # Step 5: Calculate final score
        final_score, breakdown = calculate_final_score(scores)
        
        # Save results
        save_results(scores, final_score, breakdown)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print("=" * 60)
        print(f"Final Customer Service Score: {final_score:.3f} / 1.000")
        
        return {
            'final_score': final_score,
            'breakdown': breakdown,
            'scores': scores
        }
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_audio_file>")
        print("\nExample:")
        print("  python main.py audio/call_recording.wav")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found: {audio_file_path}")
        sys.exit(1)
    
    main(audio_file_path)

