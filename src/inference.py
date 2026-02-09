"""
Inference script for Rice Disease and Nutrient Deficiency Detection

Usage:
    python inference.py --model models/MTL_ViT_Complete.pth --image path/to/leaf.jpg
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MultiTaskViT


DISEASE_NAMES = [
    'Bacterial Blight', 'Brown Spot', 'Healthy',
    'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot'
]

NUTRIENT_NAMES = [
    'Nitrogen (N) Deficiency', 
    'Phosphorus (P) Deficiency', 
    'Potassium (K) Deficiency'
]


class RicePredictor:
    """Easy-to-use predictor for rice disease and nutrient deficiency"""
    
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        
        # Load model
        self.model = MultiTaskViT(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict disease and nutrient deficiency for an image"""
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            disease_logits, nutrient_logits = self.model(input_tensor)
        
        # Get probabilities
        disease_probs = F.softmax(disease_logits, dim=1)[0]
        nutrient_probs = F.softmax(nutrient_logits, dim=1)[0]
        
        # Get predictions
        disease_pred = disease_probs.argmax().item()
        nutrient_pred = nutrient_probs.argmax().item()
        
        return {
            'disease': {
                'prediction': DISEASE_NAMES[disease_pred],
                'confidence': disease_probs[disease_pred].item() * 100,
                'all_probs': {name: prob.item() * 100 for name, prob in zip(DISEASE_NAMES, disease_probs)}
            },
            'nutrient': {
                'prediction': NUTRIENT_NAMES[nutrient_pred],
                'confidence': nutrient_probs[nutrient_pred].item() * 100,
                'all_probs': {name: prob.item() * 100 for name, prob in zip(NUTRIENT_NAMES, nutrient_probs)}
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Rice Disease & Nutrient Deficiency Prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    # Create predictor and predict
    predictor = RicePredictor(args.model)
    results = predictor.predict(args.image)
    
    # Print results
    print("\n" + "=" * 50)
    print("🌾 PREDICTION RESULTS")
    print("=" * 50)
    
    print(f"\n🔬 Disease Classification:")
    print(f"   Prediction: {results['disease']['prediction']}")
    print(f"   Confidence: {results['disease']['confidence']:.1f}%")
    print(f"\n   All probabilities:")
    for name, prob in results['disease']['all_probs'].items():
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        print(f"   {name:<20} {bar} {prob:.1f}%")
    
    print(f"\n🧪 Nutrient Deficiency:")
    print(f"   Prediction: {results['nutrient']['prediction']}")
    print(f"   Confidence: {results['nutrient']['confidence']:.1f}%")
    print(f"\n   All probabilities:")
    for name, prob in results['nutrient']['all_probs'].items():
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        print(f"   {name:<25} {bar} {prob:.1f}%")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
