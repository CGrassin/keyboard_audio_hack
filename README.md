# Breaking passwords with a microphone

This repository contains a Python proof-of-concept for breaking passwords with a microphone, using machine learning.

*Because keyboards are mechanical devices, each key may create a slightly different sound due to various manufacturing considerations. The fact that keys make a somewhat unique sound is a vulnerability. Although it is not easily picked up by our ear, it can be exploited by an algorithm...*

Please have a look at my original article here: http://charleslabs.fr/en/project-Breaking+Passwords+with+a+Microphone

## Requirements

* Python3
* Keras and Tensorflow (`pip3 install keras tensorflow`)
* argparse (`pip3 install argparse`)

## Use instructions

**Disclaimer:** this is research code, build as a proof-of-concept. It is not meant to be a practical application.

This repository includes two executable Python files:
* **split_audio.py**, a script that breaks up an audio recording file in WAV format into individual files for each key presses. It is used to generate the train data.
* **audio_reco.py**, a script that actually performs the key recognition. Several methods are included.

To generate the train data, call the "split_audio.py" script:
```bash
./split_audio.py --input ./path/to/file_with_KEY_presses.wav --out-dir ./path/to/train --label KEY
```

To launch the learning process, save the model and make a prediction:
```bash
./audio_reco.py --train-path ./path/to/train --test-path ./path/to/test.wav --model ../path/to/save/trained_model.h5
```

You may want to use the `--help` option on both scripts.