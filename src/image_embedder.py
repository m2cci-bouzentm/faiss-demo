import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class ImageEmbedder:
    _model_name = "openai/clip-vit-large-patch14"

    def __init__(self):
        torch.set_num_threads(1)
        self.model = CLIPModel.from_pretrained(ImageEmbedder._model_name)
        self.processor = CLIPProcessor.from_pretrained(ImageEmbedder._model_name)
        self.model.to('cpu')
        self.model.eval()
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            result = image_features.detach().cpu().numpy()
            return result[0]
    
    def get_embedding_dimension(self) -> int:
        return self.model.config.projection_dim

