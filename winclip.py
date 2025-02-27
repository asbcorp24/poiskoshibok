from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, image_tensor, transform=None):
        self.image_tensor = image_tensor
        self.transform = transform

    def __len__(self):
        # Return the number of items in the dataset
        # Since we have only one image, return 1
        return 1

    def __getitem__(self, idx):
        # Return the image tensor
        image = self.image_tensor
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0)
        return image
    

def winclip_load_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)

    dataset = CustomImageDataset(image_tensor)
    
    return dataset