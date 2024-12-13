import tensorflow as tf
import numpy as np
import cv2
import yt_dlp
import os
from scipy.spatial.distance import euclidean
import kagglehub

# Global flag to control the loop
running = False

# Function to get the best video stream URL
def get_best_stream_url(youtube_url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Get the best video and audio
        'noplaylist': True,                    # Only download a single video, not a playlist
        'quiet': True,                         # Suppress output
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_url = info_dict['url']
        return video_url

# Function to download a specific portion of the video
def download_video_segment(youtube_url, output_path='exercise_videos.mp4', start_time='00:00:40', duration='00:08:43', resolution='1366x768'):
    ydl_opts = {
        'format': 'bestvideo+bestaudio',
        'outtmpl': output_path,
        'quiet': True,
        'merge_output_format': 'mp4',  # Ensure final format is mp4
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# Draw keypoints on frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Draw edges and connections
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# Function to calculate similarity score
def calculate_similarity(keypoints1, keypoints2):
    # Flatten keypoints and compute Euclidean distance
    keypoints1_flat = keypoints1.flatten()
    keypoints2_flat = keypoints2.flatten()
    return np.linalg.norm(keypoints1_flat - keypoints2_flat)

# Load posture detection model
def load_model():
    kagglehub.login()
    path = kagglehub.model_download("google/movenet/tfLite/singlepose-lightning")
    
    if os.path.exists(path):
        print("Model file exists at:", path)
    else:
        print("Model file not found. Exiting.")
        exit()

    model_path = os.path.join(path, '3.tflite')
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Interpreter successfully loaded!")
        return interpreter
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()

# Setup video capture for webcam and video
def setup_video_capture(video_url):
    cap_webcam = cv2.VideoCapture(0)
    
    # Download video if not already present
    download_video_segment(video_url)
    cap_video = cv2.VideoCapture('exercise_videos.mp4')

    if not cap_webcam.isOpened():
        print("Error: Webcam not opened.")
        return None, None

    if not cap_video.isOpened():
        print("Error: Video file not opened.")
        return None, None

    return cap_webcam, cap_video

# Posture detection main loop
def run_posture_detection_system():
    video_url = "https://youtu.be/FvEb4osF0Xw?si=nlVRMGuKAMN59zOX"
    
    interpreter = load_model()
    cap_webcam, cap_video = setup_video_capture(video_url)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while cap_webcam.isOpened() and cap_video.isOpened():
        ret_webcam, frame_webcam = cap_webcam.read()
        ret_video, frame_video = cap_video.read()

        if not ret_webcam or not ret_video:
            break

        # Pre-process the frames
        img_webcam = tf.image.resize_with_pad(np.expand_dims(frame_webcam, axis=0), 192, 192)
        img_video = tf.image.resize_with_pad(np.expand_dims(frame_video, axis=0), 192, 192)

        input_image_webcam = tf.cast(img_webcam, dtype=tf.float32)
        input_image_video = tf.cast(img_video, dtype=tf.float32)

        # Make predictions for webcam
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image_webcam))
        interpreter.invoke()
        keypoints_webcam = interpreter.get_tensor(output_details[0]['index'])

        # Make predictions for video
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image_video))
        interpreter.invoke()
        keypoints_video = interpreter.get_tensor(output_details[0]['index'])

        # Draw keypoints and connections
        draw_connections(frame_webcam, keypoints_webcam, EDGES, 0.4)
        draw_keypoints(frame_webcam, keypoints_webcam, 0.4)

        draw_connections(frame_video, keypoints_video, EDGES, 0.4)
        draw_keypoints(frame_video, keypoints_video, 0.4)

        # Calculate similarity score
        similarity_score = calculate_similarity(keypoints_webcam, keypoints_video)
        print(f'Similarity Score: {similarity_score}')

        # Alert if similarity is low
        if similarity_score < 1:  # Define YOUR_THRESHOLD
            print("Alert: Posture similarity is low!")

        # Display frames
        cv2.imshow('Webcam Feed', frame_webcam)
        cv2.imshow('Exercise Video', frame_video)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_webcam.release()
    cap_video.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     video_url = "https://youtu.be/FvEb4osF0Xw?si=nlVRMGuKAMN59zOX"
    
#     interpreter = load_model()
#     cap_webcam, cap_video = setup_video_capture(video_url)
    
#     if cap_webcam and cap_video:
#         run_posture_detection_system(interpreter, cap_webcam, cap_video)
while True:
    run_posture_detection_system()
