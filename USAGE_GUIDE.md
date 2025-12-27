# Customer Service Score Predictor - Usage Guide

## Overview

This system evaluates customer service calls by analyzing various metrics including attention, empathy, greetings, ownership, interruptions, satisfaction, and talk-to-listen ratio.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys

Create or update `key.env` file with your AssemblyAI API key:

```
ASSEMBLY_AI_KEY=your_api_key_here
```

### 3. Start Ollama (Required for Speaker Classification)

Make sure Ollama is running locally with the `llama3` model:

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3
ollama serve
```

## Usage

### Quick Start

Run the complete evaluation pipeline:

```bash
python main.py path/to/your/audio_file.wav
```

### Step-by-Step Workflow

The `main.py` script automatically runs through these steps:

1. **Transcription**: Converts audio to text with speaker diarization using AssemblyAI
2. **Speaker Classification**: Identifies Customer vs Agent using Ollama
3. **Utterance Separation**: Separates customer and agent utterances
4. **Metric Calculation**: Calculates all evaluation metrics
5. **Final Score**: Computes weighted composite score

### Output

The script generates:
- Console output with progress and scores
- `transcript_output.json`: Full transcript with diarization
- `evaluation_results.json`: Complete evaluation results

## Evaluation Metrics

### 1. Attention Score (0-1)
- **Keyword Matching**: Measures overlap of keywords between customer and agent
- **Semantic Similarity**: Uses embeddings to measure semantic alignment
- **Overall**: Average of both metrics

### 2. Empathy Score (0-3)
- **Emotion Recognition**: Agent recognizes customer emotions
- **Emotion Validation**: Agent validates customer feelings
- **Support Intent**: Agent shows supportive intent
- **Final**: Sum of all three dimensions

### 3. Greetings Score (0-1)
- Measures if agent properly greets the customer in the first 5 utterances

### 4. Ownership Score (0-1)
- Measures agent's ownership and responsibility-taking language

### 5. Interruptions Score (0 or 1)
- 0 = No interruptions detected
- 1 = Interruptions detected (negative indicator)

### 6. Satisfaction Score (0-1)
- **Explicit**: Detects explicit satisfaction statements
- **Implicit**: Detects implicit satisfaction through sentiment and patterns
- **Final**: Average of both

### 7. Talk-to-Listen Ratio (0 or 1)
- 0 = Unhealthy ratio (customer or agent dominating)
- 1 = Healthy balance (0.3-0.7 ratio)

## Final Composite Score

The final score is a weighted average of all metrics:

- Attention: 15%
- Empathy: 20%
- Greetings: 10%
- Ownership: 15%
- Interruptions: 10%
- Satisfaction: 20%
- Talk-to-Listen: 10%

**You can adjust these weights in `main.py` in the `calculate_final_score()` function.**

## Programmatic Usage

You can also use the modules programmatically:

```python
from transcription_pipeline import AudioTranscription
from evaluate import Normalize_attention, Empathy, Satisfaction
# ... import other functions as needed

# Transcribe audio
transcriber = AudioTranscription(api_key="your_key")
upload_url = transcriber.upload_audio("audio.wav")
transcription_id = transcriber.perform_transcription(upload_url)
transcript_json = transcriber.get_transcript(transcription_id, "output.json")

# Classify speakers
dialogue_string = transcriber.string_4_speaker_Classification(transcript_json)
from Transcript_actions.Speaker_classification import find_speaker
speaker_classification = find_speaker(dialogue_string)

# Calculate metrics
# ... (see main.py for complete example)
```

## File Structure

```
.
├── main.py                          # Main integration script
├── evaluate.py                      # Evaluation metric functions
├── transcription_pipeline.py        # Audio transcription class
├── requirements.txt                 # Python dependencies
├── key.env                          # API keys (create this)
├── Evaluation_metrics/
│   ├── Attention.py                 # Attention scoring
│   ├── Empathy.py                   # Empathy evaluation
│   ├── Greetings_ownership.py       # Greetings & ownership
│   ├── Interruption.py              # Interruption detection
│   ├── satisfaction.py              # Satisfaction analysis
│   └── Talk_to_listen.py            # Talk-to-listen ratio
└── Transcript_actions/
    └── Speaker_classification.py    # Speaker identification
```

## Troubleshooting

### "Ollama connection error"
- Make sure Ollama is running: `ollama serve`
- Verify llama3 model is installed: `ollama list`

### "AssemblyAI API error"
- Check your API key in `key.env`
- Verify you have sufficient credits

### "spacy model not found"
- Run: `python -m spacy download en_core_web_sm`

### "Module not found"
- Install dependencies: `pip install -r requirements.txt`

## Customization

### Adjusting Weights

Edit `main.py` → `calculate_final_score()` function to change metric weights.

### Changing Portion for Satisfaction

Edit `evaluate.py` → `Satisfaction()` function default `portion` parameter.

### Modifying Greeting Patterns

Edit `Evaluation_metrics/Greetings_ownership.py` → `CANONICAL_GREETINGS` list.

## Notes

- Audio files should be in formats supported by AssemblyAI (WAV, MP3, etc.)
- Processing time depends on audio length and API response times
- Make sure you have sufficient AssemblyAI credits for transcription

