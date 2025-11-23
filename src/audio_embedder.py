import torch
import numpy as np
import soundfile as sf
from scipy import signal
from transformers import ClapProcessor, ClapModel


class AudioEmbedder:
    _model_name = "laion/clap-htsat-unfused"
    _target_sample_rate = 48000

    def __init__(self):
        torch.set_num_threads(1)
        self.model = ClapModel.from_pretrained(AudioEmbedder._model_name)
        self.processor = ClapProcessor.from_pretrained(AudioEmbedder._model_name)
        self.model.to('cpu')
        self.model.eval()
    
    def embed_audio(self, audio_path: str) -> np.ndarray:
        audio_array, sample_rate = sf.read(audio_path)
        
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        
        if sample_rate != self._target_sample_rate:
            num_samples = int(len(audio_array) * self._target_sample_rate / sample_rate)
            audio_array = signal.resample(audio_array, num_samples)
            sample_rate = self._target_sample_rate
        
        with torch.no_grad():
            inputs = self.processor(audio=audio_array, return_tensors="pt", sampling_rate=sample_rate)
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            audio_features = self.model.get_audio_features(**inputs)
            
            audio_features = audio_features / audio_features.norm(p=2, dim=-1, keepdim=True)
            
            result = audio_features.detach().cpu().numpy()
            return result[0]
    
    def get_embedding_dimension(self) -> int:
        return self.model.config.projection_dim

