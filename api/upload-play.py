from fastapi import FastAPI, File, UploadFile, HTTPException, responses
from pathlib import Path
from .main import metrics_calculation, Metrics, Final_score, load_api_key
from .LLM import Summarize
from pydantic import BaseModel, Field
from typing import Optional
import logging
import uuid
import tempfile

TEMP_DIR=Path('Upload')
TEMP_DIR.mkdir(exist_ok=True)

class Evaluation(BaseModel):
    attention_score : float
    empathy_score : float
    greet_score : float 
    ownership_score : float
    interuption_score : float
    satisfaction_score : float
    Talk_to_listen : float

class Listen(BaseModel):
    Listening : int

class Root3(BaseModel):
    Score : float=Field(..., ge=0.0, le=1.0)
    message : Optional[str]

class Root2(BaseModel):
    Attention : Root3
    Empathy : Root3
    Greet : Root3
    Ownership : Root3
    Interruption : Root3
    Satisfaction : Root3
    Listening : Listen

class Root1(BaseModel):
    Metrics : Root2
    Breakdown :  Evaluation
    Final_Score : float
    LLM_Response : str
    Satisfaction_trajectory : str

Audio_id={}

app=FastAPI()

@app.post('/Upload-audio')
async def upload(
    file : UploadFile = File(..., description='Please upload your audio clip')):

    file_suffix=Path(file.filename).suffix.lower()

    allowed_extensions={'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.mp4'}
    if file_suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f'{file.filename} not an audio file')
    
    with tempfile.NamedTemporaryFile(mode='wb',suffix= file_suffix, dir=TEMP_DIR, delete=False) as temp_file: #wb represents write in bytes
        temp_file.write(await file.read())
        temp_path=Path(temp_file.name)
        
    #In With block the tempfile is written in OS, as soon as it is closed the data is written in the disk that is in the TEMP_DIR
    #Inorder to write the data in the disk we need to call temp_file.close() but in with block as soon as it is over temp_file.close()
    # is called by itself 

    audio_unique_id=str(uuid.uuid4())
    Audio_id[audio_unique_id]= temp_path
    return {
        'audio_id' : audio_unique_id,
        'play_url' : f'/play-audio/{audio_unique_id}'
    }


@app.get('/play-audio/{audio_unique_id}')
async def play_audio(audio_unique_id : str):
    file_path=Audio_id[audio_unique_id]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail='audio clip not found')
    
    return responses.FileResponse(
        path=file_path
    )


@app.get('/Evaluate/{audio_unique_id}')
async def metrics(audio_unique_id : str):
    file_path=str(Audio_id[audio_unique_id])
    try:
        API_KEY=load_api_key()
        metrics_strings=metrics_calculation(file_path)[0]
        LLM_response=Summarize(metrics_strings)
      
        evaluation_dict=Metrics(API_KEY, file_path)[0]
        final_score=Final_score(evaluation_dict)

        response = Root1(
            Metrics = Root2(
                Attention = Root3(
                    Score = evaluation_dict['attention_score'],
                    message = metrics_strings['Attention_string']
                ),
                Empathy = Root3(
                    Score = evaluation_dict['empathy_score'],
                    message = metrics_strings['Empathy_string']
                ),
                Greet = Root3(
                    Score = evaluation_dict['greet_score'],
                    message = metrics_strings['Greeting_string']
                ),
                Ownership = Root3(
                    Score = evaluation_dict['ownership_score'],
                    message = metrics_strings['Ownership_string']
                ),
                Interruption=Root3(
                    Score = evaluation_dict['interuption_count'],
                    message = metrics_strings['Interruption_string'] 
                ),
                Satisfaction= Root3(
                    Score = evaluation_dict['satisfaction_score'],
                    message = metrics_strings['Satisfaction_string']
                ),
                Listening = Listen(
                    Score = evaluation_dict['Talk_to_Listen']
                )
            ),
            Breakdown = Evaluation(**evaluation_dict),
            Final_Score =  float(final_score),
            LLM_response = str(LLM_response),
            Satisfaction_trajectory = f'''/get_satisfaction_trajectory/{audio_unique_id}'''
            )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="internal server error")
      

@app.get('/get_satisfaction_trajectory/{audio_unique_id}')
async def send_image(audio_unique_id : str):
    file_path=str(Audio_id[audio_unique_id])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    try: 
        image_buf = metrics_calculation(file_path)[1]
        return responses.FileResponse(image_buf)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Unidentified error {type(e).__name__} occurred. Internal Server Error')


@app.delete('/delete-audio/{audio_unique_id}')
async def delete_audio(audio_unique_id: str):
    path=Audio_id.pop(audio_unique_id, None)

    if path and path.exists():
        path.unlink()

    return {'Deleted' : True}