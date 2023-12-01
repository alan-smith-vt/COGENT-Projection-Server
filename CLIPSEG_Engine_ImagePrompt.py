from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
from torch.profiler import profile, record_function, ProfilerActivity

def cropSquare(image):
    # Get image size
    width, height = image.size

    # Find the smaller dimension
    smaller_dim = min(width, height)

    # Calculate the area to crop
    left = (width - smaller_dim)/2
    top = (height - smaller_dim)/2
    right = (width + smaller_dim)/2
    bottom = (height + smaller_dim)/2

    # Crop the image
    image = image.crop((left, top, right, bottom))

    # Show the cropped image
    return image

class CLIPSEG_Engine_ImagePrompt():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model and processor
        self.processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.to(self.device)  # Move the model to GPU
        
        # Set the model to evaluate mode
        self.model.eval()
        
    def main(self, target, prompts):
        #crop the image to square
        target = cropSquare(target)
        
        # Encode the input data
        encoded_image = self.processor(images=[target], return_tensors="pt").to(self.device)  # Move the data to GPU
        encoded_prompts = [self.processor(images=[prompt], return_tensors="pt").to(self.device) for prompt in prompts]  # Move the data to GPU
                
        with torch.no_grad():
            outputs = [self.model(**encoded_image, conditional_pixel_values=encoded_prompt.pixel_values) for encoded_prompt in encoded_prompts]

        # post process result
        preds = [output.logits.unsqueeze(1) for output in outputs]
        preds = [torch.transpose(pred, 0, 1) for pred in preds]

        # Move the tensor back to CPU for visualization
        preds = [pred.cpu() for pred in preds]    

        return preds
