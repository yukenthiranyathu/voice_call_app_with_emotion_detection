# audio_processing.py
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

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

# Function to process audio, split into chunks, and return the largest chunk
def process_audio(file_path):
    sound = AudioSegment.from_file(file_path)  # Load any audio format
    
    # Split audio into chunks based on silence
    audio_chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-40)
    
    # Find the chunk with the maximum duration
    max_chunk = max(audio_chunks, key=lambda chunk: len(chunk))

    # Calculate the average frequency of the chunk
    average_frequency = calculate_average_frequency(max_chunk)

    return max_chunk, average_frequency
