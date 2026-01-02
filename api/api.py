
import os 
import json
import tempfile
from api.main import Metrics, load_api_key, Final_score
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path

app=FastAPI()

class Evaluation(BaseModel):
    attention_score : float
    empathy_sore : float
    greet_score : float 
    ownership_score : float
    interuption_score : float
    satisfaction_score : float
    Talk_to_listen : float

class Breakdown(BaseModel):
    attention : float
    empathy : float
    interuption : float
    satisfaction : float
    listening : float
    greet : bool
    ownership : bool

class Final_Output(BaseModel):
    final_agent_breakdown : float
    breakdown : Breakdown
    individual_score : Evaluation


@app.post('/evaluate', responsemodel= Final_Output)
async def Evaluate_score(
    background: BackgroundTasks,
    file : UploadFile=File(..., description='Calculate the final evaluation dictionary')):


    #Uploading audio
    allowed_extensions={'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.mp4'}
    extension=os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file type uploaded {extension} /n Allowed Extensions = {allowed_extensions}')
    
    Temp_Dir=Path('temp_upload')

    temp_path=None
    try:
        #create temporary path file
        with tempfile.NamedTemporaryFile(dir=Temp_Dir, delete=False, suffix=extension) as temp_file:
            temp_path= temp_file.name
            content = await file.read()
            temp_file.write(content)
       
        api_key=load_api_key()
        Evaluation_dictionary = Metrics(API_key=api_key, temp_path1=temp_path)
        final_score=Final_score(Evaluation_dict=Evaluation_dictionary)
        
        response = Final_Output(
            final_agent_breakdown=final_score['Final Agent Score'],
            breakdown=Breakdown(**final_score['Breakdown']),
            individual_score=Evaluation(**Evaluation_dictionary)
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f'Unexpected Error occurred : {str(e)}'
        )

if __name__ ==  '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)


    
