import os
import torch
import sounddevice as sd
import soundfile as sf

class SileroTTS:
    def __init__(self, language, model, speaker):
        self.language = language
        self.model = model
        self.speaker = speaker
        self.device = torch.device('cpu')
        self.sample_rate = 48000

        torch.set_num_threads(12)
        local_file = f'./models/silero/{language}.pt'
        
        if not os.path.isfile(local_file):
            torch.hub.download_url_to_file(f'https://models.silero.ai/models/tts/{language}/{model}.pt', local_file)

        self.model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        self.model.to(self.device)

    def generate_audio(self, text):
        audio_paths = self.model.save_wav(text=text, speaker=self.speaker, sample_rate=self.sample_rate)
        return audio_paths

    def generate_and_play_audio(self, text):
        audio_paths = self.generate_audio(text)
        
        # Use 'soundfile' to read the audio file
        audio_data, fs = sf.read(audio_paths)

        # Play the audio data
        sd.play(audio_data, fs)
        sd.wait()  # Wait until the audio is finished playing

