
import gradio as gr
from transformers import pipeline

# Load the Whisper Small ASR model from Hugging Face Transformers
asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe(audio):
    # Transcribe the audio using the Whisper model
    transcription = asr_model(audio)["text"]
    return transcription

# Gradio Interface
interface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small ASR",
    description="Automatic Speech Recognition using OpenAI Whisper Small model"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
