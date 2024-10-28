# app.py
import sys
import os
import tempfile
from audio_processing import process_audio
from emotion_prediction import load_tflite_model, predict_emotion_tflite

if __name__ == "__main__":
    # Get the audio file path from the command line argument
    audio_file_path = sys.argv[1]

    # Load your TFLite model (make sure to set the correct path)
    tflite_model_path = 'assets/speech_emotion_model.tflite'  # Update this with the actual path to your TFLite model
    interpreter = load_tflite_model(tflite_model_path)

    # Process the audio and get the predicted emotion and average frequency
    max_chunk, average_frequency = process_audio(audio_file_path)
    
    # Save the max_chunk to a temporary file in .wav format for MFCC extraction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        max_chunk.export(temp_wav.name, format="wav")
        
        # Predict emotion from the chunk with the maximum duration
        predicted_emotion = predict_emotion_tflite(interpreter, temp_wav.name)

    # Clean up the temporary file
    os.remove(temp_wav.name)

    # Print the predicted emotion and average frequency (output for Flutter app)
    print(f"Predicted Emotion: {predicted_emotion}, Average Frequency: {average_frequency}")
