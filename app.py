
import os
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import traceback

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

# Load environment variables
load_dotenv('key.env')

# Initialize FastAPI app
app = FastAPI(
    title="Customer Service Evaluation API",
    description="API for evaluating customer service calls using AI-powered metrics",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AttentionScore(BaseModel):
    matched_score: float
    similarity_score: float
    overall_attention: float


class EmpathyScore(BaseModel):
    emotion_recognition: float
    emotion_validation: float
    support_intent: float
    final_empathy_score: float


class EvaluationScores(BaseModel):
    attention: AttentionScore
    empathy: float
    greetings: float
    ownership: float
    interruptions: float
    satisfaction: float
    talk_to_listen: float


class ScoreBreakdown(BaseModel):
    attention: float
    empathy: float
    greetings: float
    ownership: float
    interruptions: float
    satisfaction: float
    talk_to_listen: float


class EvaluationResponse(BaseModel):
    final_score: float
    score_breakdown: ScoreBreakdown
    detailed_scores: EvaluationScores
    message: str = "Evaluation completed successfully"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# Helper Functions
def load_api_key():
    """Load API key from environment file."""
    api_key = os.getenv('ASSEMBLY_AI_KEY')
    if not api_key:
        raise ValueError("ASSEMBLY_AI_KEY not found in key.env file")
    return api_key


def process_audio_file(audio_path: str):
    """
    Process audio file through the complete evaluation pipeline.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with all evaluation results
    """
    try:
        # Load API key
        api_key = load_api_key()
        
        # Step 1: Transcribe audio
        transcriber = AudioTranscription(api_key=api_key)
        upload_url = transcriber.upload_audio(audio_path)
        transcription_id = transcriber.perform_transcription(upload_url)
        transcript_json = transcriber.get_transcript(transcription_id, "temp_transcript.json")
        
        # Step 2: Classify speakers
        dialogue_string = transcriber.string_4_speaker_Classification(transcript_json)
        speaker_classification = find_speaker(dialogue_string)
        corrected_utterances = corrected_list(transcript_json.copy(), speaker_classification)
        dialogue_diarized_string = String_4_Semantic_analysis(transcript_json, speaker_classification)
        
        # Step 3: Separate utterances
        customer_list, customer_string = customer_list_dict(corrected_utterances)
        agent_list, agent_string = agent_list_dict(corrected_utterances)
        
        # Step 4: Calculate all metrics
        scores = {}
        
        # Attention Score
        attention_dict = Normalize_attention(
            customer_utterance_string=customer_string,
            agent_utterance_string=agent_string,
            customer_utterance_list=customer_list,
            agent_utterance_list=agent_list
        )
        scores['attention'] = attention_dict
        
        # Empathy Score
        empathy_score = Empathy(dialogue_diarized_string)
        scores['empathy'] = empathy_score
        
        # Greetings and Ownership
        greet_score, ownership_score = Greet_Ownership(agent_list)
        scores['greetings'] = greet_score
        scores['ownership'] = ownership_score
        
        # Interruptions
        interruption_score = Interuptions(transcript_json, dialogue_string)
        scores['interruptions'] = interruption_score
        
        # Satisfaction
        satisfaction_score = Satisfaction(customer_list, portion=0.3)
        scores['satisfaction'] = satisfaction_score
        
        # Talk-to-Listen Ratio
        talk_listen_score = Talk_to_listen_ratio(dialogue_string, transcript_json)
        scores['talk_to_listen'] = talk_listen_score
        
        # Step 5: Calculate final composite score
        attention_score = scores['attention']['overall_attention']
        empathy_score_normalized = scores['empathy'] / 3.0  # Normalize from 0-3 to 0-1
        greetings_score = scores['greetings']
        ownership_score = scores['ownership']
        interruption_score_normalized = 1 - scores['interruptions']  # Invert (0=good, 1=bad)
        satisfaction_score = scores['satisfaction']
        talk_listen_score = scores['talk_to_listen']
        
        # Weighted average
        weights = {
            'attention': 0.15,
            'empathy': 0.20,
            'greetings': 0.10,
            'ownership': 0.15,
            'interruptions': 0.10,
            'satisfaction': 0.20,
            'talk_to_listen': 0.10
        }
        
        final_score = (
            attention_score * weights['attention'] +
            empathy_score_normalized * weights['empathy'] +
            greetings_score * weights['greetings'] +
            ownership_score * weights['ownership'] +
            interruption_score_normalized * weights['interruptions'] +
            satisfaction_score * weights['satisfaction'] +
            talk_listen_score * weights['talk_to_listen']
        )
        
        breakdown = {
            'attention': attention_score,
            'empathy': empathy_score_normalized,
            'greetings': greetings_score,
            'ownership': ownership_score,
            'interruptions': interruption_score_normalized,
            'satisfaction': satisfaction_score,
            'talk_to_listen': talk_listen_score
        }
        
        return {
            'final_score': round(final_score, 4),
            'score_breakdown': breakdown,
            'detailed_scores': scores
        }
        
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Service Evaluation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /evaluate": "Upload audio file for evaluation",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Customer Service Evaluation API"
    }


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to evaluate (supports common audio formats)")
):
    """
    Evaluate a customer service call audio file.
    
    This endpoint:
    1. Accepts an audio file upload
    2. Transcribes the audio using AssemblyAI
    3. Classifies speakers (Customer vs Agent)
    4. Calculates multiple evaluation metrics
    5. Returns a comprehensive evaluation report
    
    **Metrics Calculated:**
    - **Attention**: How well the agent pays attention to customer concerns
    - **Empathy**: Agent's emotional recognition, validation, and support intent
    - **Greetings**: Whether the agent properly greets the customer
    - **Ownership**: Agent's sense of responsibility and ownership
    - **Interruptions**: Whether the agent interrupts the customer
    - **Satisfaction**: Customer satisfaction indicators
    - **Talk-to-Listen Ratio**: Balance between agent talking and listening
    
    **Returns:**
    - Final composite score (0-1)
    - Individual metric scores
    - Detailed score breakdown
    """

    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.mp4'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    temp_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_path = temp_file.name
            # Write uploaded file content
            content = await file.read()
            temp_file.write(content)
        
        # Process the audio file
        results = process_audio_file(temp_path)
        
        # Clean up temporary file in background
        background_tasks.add_task(os.remove, temp_path)
        
        # Format response
        response = EvaluationResponse(
            final_score=results['final_score'],
            score_breakdown=ScoreBreakdown(**results['score_breakdown']),
            detailed_scores=EvaluationScores(
                attention=AttentionScore(**results['detailed_scores']['attention']),
                empathy=results['detailed_scores']['empathy'],
                greetings=results['detailed_scores']['greetings'],
                ownership=results['detailed_scores']['ownership'],
                interruptions=results['detailed_scores']['interruptions'],
                satisfaction=results['detailed_scores']['satisfaction'],
                talk_to_listen=results['detailed_scores']['talk_to_listen']
            ),
            message="Evaluation completed successfully"
        )
        
        return response
        
    except ValueError as e:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        error_detail = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )


@app.post("/evaluate/transcript")
async def evaluate_from_transcript(transcript_data: dict):
    """
    Evaluate a customer service call from a pre-transcribed transcript.
    
    This endpoint accepts a transcript JSON (in AssemblyAI format) and
    calculates evaluation metrics without requiring audio transcription.
    
    **Request Body:**
    - Must be a valid AssemblyAI transcript JSON format with 'utterances' field
    
    **Returns:**
    - Same format as /evaluate endpoint
    """
    try:
        # Validate transcript format
        if 'utterances' not in transcript_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid transcript format. Must include 'utterances' field."
            )
        
        transcript_json = transcript_data
        
        # Step 2: Classify speakers
        transcriber = AudioTranscription(api_key="dummy")
        dialogue_string = transcriber.string_4_speaker_Classification(transcript_json)
        speaker_classification = find_speaker(dialogue_string)
        corrected_utterances = corrected_list(transcript_json.copy(), speaker_classification)
        dialogue_diarized_string = String_4_Semantic_analysis(transcript_json, speaker_classification)

        customer_list, customer_string = customer_list_dict(corrected_utterances)
        agent_list, agent_string = agent_list_dict(corrected_utterances)
        
        scores = {}
        
        # Attention Score
        attention_dict = Normalize_attention(
            customer_utterance_string=customer_string,
            agent_utterance_string=agent_string,
            customer_utterance_list=customer_list,
            agent_utterance_list=agent_list
        )
        scores['attention'] = attention_dict
        
        # Empathy Score
        empathy_score = Empathy(dialogue_diarized_string)
        scores['empathy'] = empathy_score
        
        # Greetings and Ownership
        greet_score, ownership_score = Greet_Ownership(agent_list)
        scores['greetings'] = greet_score
        scores['ownership'] = ownership_score
        
        # Interruptions
        interruption_score = Interuptions(transcript_json, dialogue_string)
        scores['interruptions'] = interruption_score
        
        # Satisfaction
        satisfaction_score = Satisfaction(customer_list, portion=0.3)
        scores['satisfaction'] = satisfaction_score
        
        # Talk-to-Listen Ratio
        talk_listen_score = Talk_to_listen_ratio(dialogue_string, transcript_json)
        scores['talk_to_listen'] = talk_listen_score
        
        # Step 5: Calculate final composite score
        attention_score = scores['attention']['overall_attention']
        empathy_score_normalized = scores['empathy'] / 3.0
        greetings_score = scores['greetings']
        ownership_score = scores['ownership']
        interruption_score_normalized = 1 - scores['interruptions']
        satisfaction_score = scores['satisfaction']
        talk_listen_score = scores['talk_to_listen']

        weights = {
            'attention': 0.15,
            'empathy': 0.20,
            'greetings': 0.10,
            'ownership': 0.15,
            'interruptions': 0.10,
            'satisfaction': 0.20,
            'talk_to_listen': 0.10
        }
        
        final_score = (
            attention_score * weights['attention'] +
            empathy_score_normalized * weights['empathy'] +
            greetings_score * weights['greetings'] +
            ownership_score * weights['ownership'] +
            interruption_score_normalized * weights['interruptions'] +
            satisfaction_score * weights['satisfaction'] +
            talk_listen_score * weights['talk_to_listen']
        )
        
        breakdown = {
            'attention': attention_score,
            'empathy': empathy_score_normalized,
            'greetings': greetings_score,
            'ownership': ownership_score,
            'interruptions': interruption_score_normalized,
            'satisfaction': satisfaction_score,
            'talk_to_listen': talk_listen_score
        }
        
        response = EvaluationResponse(
            final_score=round(final_score, 4),
            score_breakdown=ScoreBreakdown(**breakdown),
            detailed_scores=EvaluationScores(
                attention=AttentionScore(**scores['attention']),
                empathy=scores['empathy'],
                greetings=scores['greetings'],
                ownership=scores['ownership'],
                interruptions=scores['interruptions'],
                satisfaction=scores['satisfaction'],
                talk_to_listen=scores['talk_to_listen']
            ),
            message="Evaluation completed successfully"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_detail = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing transcript: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

