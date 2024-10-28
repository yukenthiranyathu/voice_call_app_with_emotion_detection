# emotion_prediction.py
import numpy as np
import librosa
import tensorflow as tf

# Load the TensorFlow Lite model
def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc.reshape(1, -1, 1).astype(np.float32)  # Convert to float32 for TFLite model
    except Exception as e:
        raise ValueError(f"Error extracting features from {file_path}: {e}")

# Function to predict emotion using the TensorFlow Lite model
def predict_emotion_tflite(interpreter, file_path):
    try:
        # Extract MFCC features
        features = extract_mfcc(file_path)

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], features)

        # Run inference
        interpreter.invoke()

        # Get the output (predictions)
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Assuming you have 4 emotions (update labels as needed)
        emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']  # Update based on your labels
        predicted_emotion = emotion_labels[predicted_class_index]

        return predicted_emotion
    except Exception as e:
        raise ValueError(f"Failed to predict emotion: {e}")
