import os
import torch
import transformers
# import elevenlabs
import speech_recognition as sr
from queue import Queue
from accelerate import Accelerator
import pyttsx3

# --------------------------- Configuration --------------------------- #

# # Set API keys for ElevenLabs
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# response_tts = elevenlabs.ElevenLabs(
#     api_key=ELEVENLABS_API_KEY,
# )

# Initialize the TTS engine
tts_engine = pyttsx3.init()

# Initialize the recognizer
recognizer = sr.Recognizer()

# Queue to handle transcripts
transcript_queue = Queue()

# --------------------------- Load SEA-LION Model --------------------------- #

# Specify the SEA-LION model ID from HuggingFace
model_id = "aisingapore/sea-lion-3b"

# Initialize Accelerator for device and memory management
accelerator = Accelerator(mixed_precision="fp16")

# Load the tokenizer and model with trust_remote_code=True
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load model with max memory allocation split between GPU and CPU
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use mixed precision for efficiency
    device_map="auto",  # Automatically offload layers to CPU if necessary
    max_memory={0: "6GB", "cpu": "4GB"}  # Adjust GPU and CPU memory usage
)

# Prepare model and tokenizer with Accelerator for mixed precision & device management
model, tokenizer = accelerator.prepare(model, tokenizer)

# Initialize the text generation pipeline
pipeline = transformers.pipeline(
    task="text-generation",
    model=model,  # Pass the actual model object
    tokenizer=tokenizer
)

# --------------------------- Conversation Handler --------------------------- #

def handle_conversation(switch):
    while switch:
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

                # Define the prompt for SEA-LION
                prompt = f"You are a helpful assistant. {user_input}"

                # Generate a response using SEA-LION
                response = pipeline(
                    prompt,
                    max_new_tokens=75,
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

handle_conversation(True)