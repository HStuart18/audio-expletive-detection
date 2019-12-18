# audio-expletive-detection
This project aims to create a system for detecting expletives in streaming audio. Such a system should be exposable as a Web API.

Currently working on implementing trigger word detection. 10sec of background audio is overlayed with positive and negative samples. Because I have overlayed the audio, I know at which exact timestamps a trigger word (positive clip) occurs. However, the output vector is too sparse and the network converges to all zeros.
