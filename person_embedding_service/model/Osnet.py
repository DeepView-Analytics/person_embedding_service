import torch
from torchreid.utils import load_pretrained_weights
from torchreid.models import build_model
from torchreid.data.transforms import build_transforms
from PIL import Image
import io
import asyncio

class OSNetEmbedder:
    def __init__(self, model_name='osnet_x1_0', num_classes=1000):
        # Load and initialize the model
        self.model = build_model(name=model_name, num_classes=num_classes)
        load_pretrained_weights(self.model, r'D:\DeepView\MicroServices\person_embedding\person_embedding_service\model\osnet_x1_0_imagenet.pth')
        self.model.eval()
        
        # Initialize the image transformations
        self.transforms = build_transforms(height=256, width=128)[1]
    
    async def __preprocess_image(self, img):

        img_transformed = self.transforms(img)
        return img_transformed.unsqueeze(0)
    
    async def __postprocess_inference(self, features):
        """Private async method to postprocess the model's inference output."""
        return features.cpu().numpy().tolist()
    
    async def __inference(self, batch_images_tensor):
        """Private async method to run inference on a batch of preprocessed images."""
        with torch.no_grad():
            features = self.model(batch_images_tensor)
        return features
    
    async def get_embeddings(self, images):
        """Public async method to get embeddings for a list of byte arrays."""
        batch_images = []
        
        # Preprocess each image asynchronously
        preprocess_tasks = [self.__preprocess_image(image) for image in images]
        batch_images = await asyncio.gather(*preprocess_tasks)
        
        # Stack the batch of images into a single tensor
        batch_images_tensor = torch.cat(batch_images, dim=0)
        
        # Run inference
        features = await self.__inference(batch_images_tensor)
        
        # Postprocess the inference output
        embeddings = await self.__postprocess_inference(features)
        
        return embeddings

# Example usage
async def main():
    # Load sample images as byte arrays
    image_paths = [
        r'D:\DeepView\MicroServices\person_embedding\saru.jpg',
        r'D:\DeepView\MicroServices\person_embedding\1.jpg'
    ]
    
    image_byte_arrays = [open(image_path, 'rb').read() for image_path in image_paths]

    # Create an instance of the embedder
    embedder = OSNetEmbedder()
    
    # Get embeddings for the images
    embeddings = await embedder.get_embeddings(image_byte_arrays)
    print("Feature Embeddings:", embeddings)

# Run the asynchronous main function
if __name__ == '__main__':
    asyncio.run(main())
