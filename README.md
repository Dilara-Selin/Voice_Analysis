# Speaker Recognition and Sentiment Analysis Application

## Overview

This project is a **Speaker Recognition and Sentiment Analysis Application** developed using Python. The app records audio, identifies the speaker, transcribes the audio, translates it to English, and performs sentiment and topic analysis. The graphical user interface (GUI) is built using Tkinter.

## Key Features

- **Audio Recording**: Record audio directly from the microphone.
- **Speaker Recognition**: Identifies the speaker using a pre-trained deep learning model.
- **Transcription**: Converts audio to text using Google's Speech Recognition API.
- **Translation**: Translates the transcription from Turkish to English.
- **Sentiment Analysis**: Analyzes the emotional tone of the text.
- **Topic Classification**: Classifies the topic of the speech using a zero-shot classification model.
- **Visualization**: Plots the waveform and histogram of the recorded audio.

## Technologies and Libraries

- **GUI**: Tkinter
- **Audio Processing**: Sounddevice, Scipy, Librosa, Pydub
- **Machine Learning**: TensorFlow, Transformers
- **Speech Recognition**: SpeechRecognition
- **Translation**: Deep Translator
- **Visualization**: Matplotlib

## Installation

### Prerequisites

Ensure the following are installed:

- Python 3.8+
- pip

### Required Packages

Install the required Python packages using the following command:

```bash
pip install numpy scipy librosa tensorflow sounddevice speechrecognition pydub matplotlib deep-translator transformers
```

### Additional Dependencies

- **FFmpeg** (for audio processing):

```bash
sudo apt install ffmpeg  # For Linux
brew install ffmpeg      # For macOS
winget install ffmpeg    # For Windows
```

## How to Run

1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Run the application:

```bash
python interface.py
```

4. Use the buttons in the GUI to start and stop recording, and process the audio.

## Project Structure

```
|-- interface.py               # Main application file
|-- ses_tanima_modeli.keras    # Pre-trained speaker recognition model
|-- mikrofon_kayit.wav         # Temporary audio recording file
|-- converted_audio.wav        # Processed audio for transcription
|-- histogram.ipynb            # Jupyter notebook for audio histogram visualization
|-- model.ipynb                # Jupyter notebook for training/testing models
|-- recording.ipynb            # Jupyter notebook for handling audio recording
```

## Usage

1. **Record Audio**: Press the "Kayıt Başlat" button to start recording.
2. **Stop Recording**: Press the "Kayıt Durdur" button.
3. **Process Recording**: Press the "Kaydı İşle" button to analyze the audio.
4. View speaker identification, transcription, sentiment, and topic analysis in the GUI.

## Customization

- **Add Speakers**: Update the `sinif_isimleri` list in `interface.py` to include more speaker names.
- **Modify Models**: Replace the pre-trained `ses_tanima_modeli.keras` with your own model.
- **Adjust Recording Length**: Modify the `self.saniye` variable to adjust recording duration.

## Potential Improvements

- Implement real-time speaker identification.
- Add language selection for transcription and translation.
- Integrate more robust emotion and topic classification models.

## Contributing

We welcome contributions to improve this project! Here are some ways you can contribute:

- **Report bugs and suggest features** via GitHub Issues.
- **Submit pull requests** for new features, bug fixes, or improvements.
- **Improve documentation and examples** to help others use and understand the project.


Please follow the Contributor Covenant Code of Conduct in all your interactions with the project.

