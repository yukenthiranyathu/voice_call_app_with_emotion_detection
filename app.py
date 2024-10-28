import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
from pydub.silence import split_on_silence
from flask import Flask, request, jsonify
import tempfile
import os
from collections import Counter

app = Flask(__name__)

# Load the TensorFlow Lite model
def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, -1, 1).astype(np.float32)

# Function to predict emotion using the TensorFlow Lite model
def predict_emotion_tflite(interpreter, file_path):
    features = extract_mfcc(file_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions)
    emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
    predicted_emotion = emotion_labels[predicted_class_index]
    return predicted_emotion

# Function to process audio, split into chunks, and predict emotions
def process_audio(interpreter, file_path):
    sound = AudioSegment.from_file(file_path)  # Load any audio format
    
    # Split audio into chunks based on silence
    audio_chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-40)

    # Collect predicted emotions for each chunk
    emotions = []
    for chunk in audio_chunks:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            chunk.export(temp_wav.name, format="wav")  # Export chunk to temporary WAV file
            predicted_emotion = predict_emotion_tflite(interpreter, temp_wav.name)
            emotions.append(predicted_emotion)

        # Clean up the temporary file
        os.remove(temp_wav.name)

    # Find the most common emotion
    most_common_emotion = Counter(emotions).most_common(1)
    if most_common_emotion:
        return most_common_emotion[0][0]  # Return the most common emotion

    return None  # Return None if no emotions were predicted

@app.route('/process_audio', methods=['POST'])
def handle_process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        tflite_model_path = 'assets/speech_emotion_model.tflite'  # Update path if necessary
        interpreter = load_tflite_model(tflite_model_path)

        # Process the audio and get the most frequent emotion
        predicted_emotion = process_audio(interpreter, temp_file.name)

    os.remove(temp_file.name)
    return jsonify({"emotion": predicted_emotion})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
