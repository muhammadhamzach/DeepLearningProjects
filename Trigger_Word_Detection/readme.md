# Trigger Word Detection

* main.py is used to run the file
* Built-in code uses the trigger word "activate" and produces a chime.
* User can use their own .wav files with activate embedded in it. The audio files should be 10sec. Extra length will be trimmed.
* If user want to change the trigger word then they can navigate to raw_data/activates and put the activiate examples in it and then uncomment the lines in main.py to create a training example from the activate files. This can then be put in a loop of m-examples and the model can be trained on it from there.

### Reference
This code is modified from the sequence modeling course on deeplearning.ai