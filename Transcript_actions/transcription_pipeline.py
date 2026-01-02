import os
import requests
import time
import json

#Initialising all the necessary variables
class AudioTranscription:
    def __init__(self, api_key: str):
        '''
        passing the necessary arguments

        ARGS : API_KEY
        '''
        self.API_KEY = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        self.headers={
            'authorization': self.API_KEY
        }
#Transcription involves the following steps: upload -> perform transcription -> check_status of transcription-> once completed json.dump
    def upload_audio(self, audio_path: str):
        '''
        To get the response on POST of upload URL

        ARGS : audio file path
        
        RETURN : To get the upload URL at Assembly AI server endpoint
        '''
        with open(audio_path, "rb") as f:
            upload_request = requests.post(
                url=f"{self.base_url}/upload",
                headers={
                    "authorization": self.API_KEY
                },
                data=f,
            )
        if upload_request.status_code!=200:
            raise Exception(f'Upload failed : {upload_request.text}')
        return upload_request.json()["upload_url"]

    def perform_transcription(self, upload_url: str):
        '''
        Sending the upload url to ASsembly AI endpoint 
        and sending request to perform transcription.

        ARGS : Assembly Ai response UPLOAD URL

        RETURN : Transcription id
        '''
        transcription = requests.post(
            url=f"{self.base_url}/transcript",
            headers={"authorization": self.API_KEY},
            json={"audio_url": upload_url, "speaker_labels": True, "speaker_options": {
      "min_speakers_expected": 2,
      "max_speakers_expected": 5}}
            )
        return transcription.json()['id']

    def get_transcript(self, transcription_id:str):
        '''
        Retrieving the transcript

        ARGS: 
        transcription_id : received from Assembly AI
        saved_file_path : path where the transcription text is to be saved
        
        RETURN : 
        Dictionary of response form assembly ai containing the details with the diazrized transcript
        '''
        start=time.time()
        while True:
            transcription_process=requests.get(
            url=f'{self.base_url}/transcript/{transcription_id}',
            headers=self.headers)


            if transcription_process.status_code!=200:
                raise RuntimeError(f'Status Check failed :{transcription_process.status_code}, {transcription_process.text}')

            status=(transcription_process.json()).get('status')


            if status=='completed':
                break

            if status=='error':
                raise RuntimeError(f'Transcription failed')

            if (time.time() - start)> 300:
                raise TimeoutError('Transcription still not completed after waiting for 300 seconds')

            else:
                print('The task is still under process.....please wait')
            time.sleep(3)
        return transcription_process.json()


    def string_4_speaker_Classification(self, transcription_process:dict):
        '''
        For converting the json of dialogues into a full readable string for the Speaker classification by Ollama
        Sample output of the transcript json after speech diarization:
        {
  "id": "abc123",
  "status": "completed",
  "language_code": "en_us",
  "confidence": 0.94,
  "audio_duration": 18.7,

  "utterances": [
    {
      "speaker": "A",
      "text": "Hello, thank you for calling customer support.",
      "start": 1200,
      "end": 3400,
      "confidence": 0.96
    },
    {
      "speaker": "B",
      "text": "Hi, I'm having an issue with my internet connection.",
      "start": 3600,
      "end": 6800,
      "confidence": 0.95
    },
    {
      "speaker": "A",
      "text": "I’m sorry to hear that. Could you please describe the problem?",
      "start": 7100,
      "end": 10200,
      "confidence": 0.94
    },
    {
      "speaker": "B",
      "text": "Yes, the connection drops every few minutes.",
      "start": 10400,
      "end": 13800,
      "confidence": 0.93
    }
  ]
}

        '''
        dialogue_string=''
        utterances_list=transcription_process.get('utterances')
        for u in utterances_list:
            dialogue_string+=f"Speaker {u['speaker']}: {u['text']}\n"

        return dialogue_string
        
#from transcription_pipeline import AudioTranscription

# transcriber = AudioTranscription(api_key="YOUR_API_KEY")

# upload_url = transcriber.upload_audio("path/to/audio.wav")
# transcription_id = transcriber.perform_transcription(upload_url)
# transcript_json = transcriber.get_transcript(transcription_id, "output.json")
# dialogue_string = transcriber.string_4_speaker_Classification(transcript_json




