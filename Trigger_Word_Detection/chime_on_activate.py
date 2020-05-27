from pydub import AudioSegment

chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Loop over the output steps in the y
    for i in range(Ty):
        # Increment consecutive output steps
        consecutive_timesteps += 1
        #  If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Reset consecutive output steps to 0
            consecutive_timesteps = 0
    
    audio_clip.export("chime_output.wav", format='wav')
    print("output file with chime (chime_output.wav) exported to the directory")