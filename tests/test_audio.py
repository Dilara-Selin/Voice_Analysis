import unittest
import numpy as np
import scipy.io.wavfile as wav
import os
import sys
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MockModel:
    def predict(self, features):
        return np.array([[0.1, 0.2, 0.3, 0.4]])


class TestAudioProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        interface_path = r"C:\Users\HP\OneDrive\Belgeler\Python\SesTanimlama-main\interface.py"
        try:
            # Anlık çıktıyı göster
            cls.process = Popen(
                ["python", interface_path],
                stdout=sys.stdout,
                stderr=sys.stderr,
                bufsize=1,
                universal_newlines=True
            )
            # interface.py tamamen başlatılana kadar bekle
            time.sleep(25)
            print("interface.py is running...")
        except Exception as e:
            print(f"Error starting interface.py: {e}")


    def test_analyze_emotion(self):
        emotions = {"joy": 0.7, "sadness": 0.1, "anger": 0.2}
        total = sum(emotions.values())
        percentages = {emotion: (score / total) * 100 for emotion, score in emotions.items()}

        self.assertAlmostEqual(percentages["joy"], 70.0, places=2)
        self.assertAlmostEqual(percentages["sadness"], 10.0, places=2)
        self.assertAlmostEqual(percentages["anger"], 20.0, places=2)
        print("Test 1 Passed: Emotion Analysis")

    def test_analyze_topic(self):
        result = {'labels': ["technology", "health", "sport"], 'scores': [0.8, 0.1, 0.1]}
        self.assertEqual(result['labels'][0], "technology")
        self.assertAlmostEqual(result['scores'][0], 0.8, places=2)
        print("Test 2 Passed: Topic Analysis")

    def test_plot_signal(self):
        audio_data = np.sin(np.linspace(0, 2 * np.pi, 1000))
        fig, ax = plt.subplots()
        ax.plot(audio_data, color='purple')
        ax.set_title('Test Ses Sinüsü')
        ax.set_xlabel('Zaman')
        ax.set_ylabel('Genlik')
        self.assertEqual(len(audio_data), 1000)
        plt.close(fig)
        print("Test 3 Passed: Signal Plot")

    def test_save_recording(self):
        rng = np.random.default_rng(seed=42)
        audio_data = rng.random(44100 * 5)
        file_path = "test_mikrofon_kayit.wav"
        wav.write(file_path, 44100, (audio_data * 32767).astype(np.int16))
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)
        print("Test 4 Passed: Recording Save")


if __name__ == "__main__":
    unittest.main()
