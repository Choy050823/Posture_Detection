import asyncio
import websockets
# from posture_detect_video import run_posture_detection_system
from voice_AI import handle_conversation

def handle_req(msg):
    print("Handling msg")
    if (msg == "Start Exercise Video"):
        print("Initiating Posture Detection System...")
        # run_posture_detection_system(True)
    elif (msg == "End Exercise Video"):
        print("Terminating Posture Detection System...")
        # run_posture_detection_system(False)
    elif (msg == "Start Voicebot"):
        print("Initiating Voicebot AI...")
        handle_conversation(True)
    elif (msg == "End Voicebot"):
        print("Terminating Voicebot AI...")
        handle_conversation(False)

async def handle_connection(websocket, path):
    print("Client connected")
    try:
        async for message in websocket:
            handle_req(message)
            print(f"Received message: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        print("Client disconnected", e)

async def main():
    async with websockets.serve(handle_connection, "10.100.118.52", 8765):
        print("WebSocket server is running on ws://10.100.118.52:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
