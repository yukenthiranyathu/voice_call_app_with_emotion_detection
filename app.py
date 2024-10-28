import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
import os

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
    return mfcc.reshape(1, -1, 1).astype(np.float32)  # Convert to float32 for TFLite model

# Function to predict emotion using the TensorFlow Lite model
def predict_emotion_tflite(interpreter, features):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions)
    
    # Update emotion labels as needed
    emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
    return emotion_labels[predicted_class_index]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file temporarily
    temp_file_path = 'temp_audio.wav'
    file.save(temp_file_path)

    try:
        # Load the model and predict emotion
        tflite_model_path = 'speech emotion detection.tflite'  # Ensure the correct path
        interpreter = load_tflite_model(tflite_model_path)

        # Extract MFCC features and predict emotion
        features = extract_mfcc(temp_file_path)
        predicted_emotion = predict_emotion_tflite(interpreter, features)

        return jsonify({'predicted_emotion': predicted_emotion}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
