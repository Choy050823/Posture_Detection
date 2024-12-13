import os
import torch
import transformers
import speech_recognition as sr
from queue import Queue
from accelerate import Accelerator
import pyttsx3
from huggingface_hub import login

# Login to Hugging Face
login("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")

# Initialize the TTS engine
tts_engine = pyttsx3.init()

# Initialize the recognizer
recognizer = sr.Recognizer()

# Queue to handle transcripts
transcript_queue = Queue()

# Specify the quantized SEA-LION model ID from HuggingFace
model_id = "aisingapore/llama3-8b-cpt-sea-lion-v2.1-instruct-gguf"

# Initialize Accelerator for device and memory management
accelerator = Accelerator(mixed_precision="fp16")

# Load the tokenizer and model with trust_remote_code=True
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "8GB", "cpu": "4GB"}
)

# Prepare model and tokenizer with Accelerator
model, tokenizer = accelerator.prepare(model, tokenizer)

# Initialize the text generation pipeline
pipeline = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer
)

# --------------------------- Conversation Handler --------------------------- #

def handle_conversation():
    while True:
        # Use the microphone as the audio source
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source)

            print("Listening for speech...")
            audio_data = recognizer.listen(source)

            try:
                # Convert speech to text using Google's speech recognition
                transcript = recognizer.recognize_google(audio_data)
                transcript_queue.put(transcript + ' ')
                print("User:", transcript)

                # Retrieve data from the queue
                user_input = transcript_queue.get()

                # Define a prompt for the SEA-LION model
                prompt = "You are a helpful assistant. How can I assist you today?"

                # Generate a response using SEA-LION
                response = pipeline(
                    prompt,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

                # Extract the generated text
                generated_text = response[0]['generated_text']
                
                # Optional: Clean the response by removing the prompt part
                if generated_text.lower().startswith(prompt.lower()):
                    generated_text = generated_text[len(prompt):].strip()

                print("\nAI:", generated_text)

                # Convert the response to audio and play it
                # audio = response_tts.generate(
                #     text=generated_text,
                #     voice="Bill"  # Use a specific voice
                # )

                # response_tts.play(audio)
                # Speak text
                tts_engine.say(generated_text)

                # Wait until speech is finished
                tts_engine.runAndWait()
                
            except sr.UnknownValueError:
                print("Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results from Speech Recognition service; {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

# --------------------------- Run the Conversation Handler --------------------------- #

if __name__ == "__main__":
    handle_conversation()
