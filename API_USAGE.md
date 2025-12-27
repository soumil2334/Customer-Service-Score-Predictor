# FastAPI Backend Usage Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have your AssemblyAI API key in `key.env`:
```
ASSEMBLY_AI_KEY=your_api_key_here
```

3. Ensure Ollama is running locally (for speaker classification and empathy analysis):
```bash
# Start Ollama service
ollama serve
```

## Running the API

### Development Server
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "Customer Service Evaluation API"
}
```

### 2. Evaluate Audio File
**POST** `/evaluate`

Upload an audio file for evaluation.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with `file` field containing the audio file

**Supported Audio Formats:**
- .wav
- .mp3
- .m4a
- .flac
- .ogg
- .webm
- .mp4

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"
```

**Example using Python requests:**
```python
import requests

url = "http://localhost:8000/evaluate"
files = {"file": open("audio.wav", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "final_score": 0.8234,
  "score_breakdown": {
    "attention": 0.85,
    "empathy": 0.80,
    "greetings": 1.0,
    "ownership": 0.75,
    "interruptions": 1.0,
    "satisfaction": 0.90,
    "talk_to_listen": 1.0
  },
  "detailed_scores": {
    "attention": {
      "matched_score": 0.82,
      "similarity_score": 0.88,
      "overall_attention": 0.85
    },
    "empathy": 2.4,
    "greetings": 1.0,
    "ownership": 0.75,
    "interruptions": 0,
    "satisfaction": 0.90,
    "talk_to_listen": 1.0
  },
  "message": "Evaluation completed successfully"
}
```

### 3. Evaluate from Transcript
**POST** `/evaluate/transcript`

Evaluate a call using a pre-transcribed transcript (AssemblyAI format).

**Request:**
- Method: POST
- Content-Type: application/json
- Body: JSON object with AssemblyAI transcript format

**Example:**
```python
import requests

url = "http://localhost:8000/evaluate/transcript"
transcript = {
  "utterances": [
    {
      "speaker": "A",
      "text": "Hello, how may I help you?",
      "start": 0,
      "end": 2000,
      "confidence": 0.95
    },
    {
      "speaker": "B",
      "text": "Hi, I'm having an issue with my account.",
      "start": 2500,
      "end": 5000,
      "confidence": 0.94
    }
  ]
}

response = requests.post(url, json=transcript)
print(response.json())
```

## Evaluation Metrics

The API calculates the following metrics:

1. **Attention** (0-1): How well the agent pays attention to customer concerns
   - `matched_score`: Keyword overlap between customer and agent
   - `similarity_score`: Semantic similarity of responses
   - `overall_attention`: Combined attention score

2. **Empathy** (0-3): Agent's emotional intelligence
   - Emotion recognition
   - Emotion validation
   - Support intent

3. **Greetings** (0-1): Whether agent properly greets customer

4. **Ownership** (0-1): Agent's sense of responsibility

5. **Interruptions** (0 or 1): Whether agent interrupts customer (0=interrupted, 1=no interruption)

6. **Satisfaction** (0-1): Customer satisfaction indicators

7. **Talk-to-Listen Ratio** (0-1): Balance between agent talking and listening (0.3-0.7 is healthy)

## Final Score Calculation

The final composite score is calculated using weighted averages:

- Attention: 15%
- Empathy: 20%
- Greetings: 10%
- Ownership: 15%
- Interruptions: 10%
- Satisfaction: 20%
- Talk-to-Listen: 10%

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid file type, missing fields, etc.)
- `500`: Internal Server Error (processing errors)

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Notes

- Audio transcription can take time depending on file size
- Make sure Ollama is running with the `llama3` model available
- The API processes files asynchronously but returns results synchronously
- Temporary files are automatically cleaned up after processing

