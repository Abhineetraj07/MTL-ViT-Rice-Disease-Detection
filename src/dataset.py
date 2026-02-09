"""
Dataset utilities for Rice Multi-Task Learning
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class RiceMultiTaskDataset(Dataset):
    """Custom Dataset for Multi-Task Learning"""
    
    DISEASE_CLASSES = [
        'bacterial_leaf_blight', 'brown_spot', 'healthy',
        'leaf_blast', 'leaf_scald', 'narrow_brown_spot'
    ]
    
    NUTRIENT_CLASSES = ['nitrogenn', 'phosphorusp', 'potassiumk']
    
    DISEASE_NAMES = [
        'Bacterial Blight', 'Brown Spot', 'Healthy',
        'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot'
    ]
    
    NUTRIENT_NAMES = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.disease_to_idx = {c: i for i, c in enumerate(self.DISEASE_CLASSES)}
        self.nutrient_to_idx = {c: i for i, c in enumerate(self.NUTRIENT_CLASSES)}
        
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        split_dir = os.path.join(self.data_dir, self.split)
        
        # Load disease images
        disease_dir = os.path.join(split_dir, 'disease')
        if os.path.exists(disease_dir):
            for class_name in os.listdir(disease_dir):
                class_path = os.path.join(disease_dir, class_name)
                if os.path.isdir(class_path) and class_name in self.disease_to_idx:
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.samples.append({
                                'path': os.path.join(class_path, img_name),
                                'disease_label': self.disease_to_idx[class_name],
                                'nutrient_label': -1,
                                'task': 'disease'
                            })
        
        # Load nutrient images
        nutrient_dir = os.path.join(split_dir, 'nutrient')
        if os.path.exists(nutrient_dir):
            for class_name in os.listdir(nutrient_dir):
                class_path = os.path.join(nutrient_dir, class_name)
                if os.path.isdir(class_path) and class_name in self.nutrient_to_idx:
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.samples.append({
                                'path': os.path.join(class_path, img_name),
                                'disease_label': -1,
                                'nutrient_label': self.nutrient_to_idx[class_name],
                                'task': 'nutrient'
                            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'disease_label': sample['disease_label'],
            'nutrient_label': sample['nutrient_label'],
            'task': sample['task']
        }


def get_transforms(split='train', image_size=224):
    """Get image transformations"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
