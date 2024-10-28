import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load the TensorFlow Lite model
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

# Extract MFCC features from an audio file
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, -1, 1).astype(np.float32)

# Predict emotion using the TensorFlow Lite model
def predict_emotion(interpreter, features):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], features)
    interpreter.invoke()
    predictions = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return ['Angry', 'Happy', 'Neutral', 'Sad'][np.argmax(predictions)]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file provided'}), 400

    temp_file_path = 'temp_audio.wav'
    request.files['file'].save(temp_file_path)

    try:
        interpreter = load_model('assets/speech_emotion_model.tflite')
        features = extract_mfcc(temp_file_path)
        predicted_emotion = predict_emotion(interpreter, features)
        return jsonify({'predicted_emotion': predicted_emotion}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(temp_file_path) if os.path.exists(temp_file_path) else None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
