import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
import os

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

# Function to calculate the average frequency of an audio segment
def calculate_average_frequency(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    
    # If stereo, take only one channel
    if audio_segment.channels == 2:
        samples = samples[::2]

    sample_rate = audio_segment.frame_rate
    
    # Perform FFT to get frequencies
    fft_result = np.fft.fft(samples)
    frequencies = np.fft.fftfreq(len(fft_result), d=1/sample_rate)
    magnitudes = np.abs(fft_result)

    # Calculate average frequency of the signal
    average_frequency = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
    
    return average_frequency

# Function to process audio, split into chunks, and predict emotion
def process_audio(interpreter, file_path):
    sound = AudioSegment.from_file(file_path)  # Load any audio format
    
    # Split audio into chunks based on silence
    audio_chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-40)
    
    # Find the chunk with the maximum duration
    max_chunk = max(audio_chunks, key=lambda chunk: len(chunk))

    # Calculate the average frequency of the chunk
    average_frequency = calculate_average_frequency(max_chunk)

    # Save the chunk to a temporary file in .wav format for MFCC extraction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        max_chunk.export(temp_wav.name, format="wav")
        
        # Predict emotion from the chunk with the maximum duration
        predicted_emotion = predict_emotion_tflite(interpreter, temp_wav.name)

    # Clean up the temporary file
    os.remove(temp_wav.name)

    return predicted_emotion,

if __name__ == "__main__":
    import sys

    # Get the audio file path from the command line argument
    audio_file_path = sys.argv[1]

    # Load your TFLite model (make sure to set the correct path)
    tflite_model_path = 'assets/speech_emotion_model.tflite'  # Update this with the actual path to your TFLite model
    interpreter = load_tflite_model(tflite_model_path)

    # Process the audio and get the predicted emotion and average frequency
    predicted_emotion = process_audio(interpreter, audio_file_path)

    # Print the predicted emotion and average frequency (output for Flutter app)
    print(f"Predicted Emotion: {predicted_emotion}")
