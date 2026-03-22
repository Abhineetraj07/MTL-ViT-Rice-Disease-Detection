import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# Model definition
class MultiTaskViT(nn.Module):
    def __init__(self, num_disease_classes=6, num_nutrient_classes=3, pretrained=False):
        super(MultiTaskViT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.embed_dim
        self.disease_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_disease_classes))
        self.nutrient_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_nutrient_classes))

    def forward(self, x):
        features = self.backbone(x)
        return self.disease_head(features), self.nutrient_head(features)

# Class names
DISEASE_NAMES = ['Bacterial Blight', 'Brown Spot', 'Healthy',
                 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']
NUTRIENT_NAMES = ['Nitrogen (N) Deficiency', 'Phosphorus (P) Deficiency',
                  'Potassium (K) Deficiency']

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskViT(pretrained=False)

# Load weights if available
import os
if os.path.exists('MTL_ViT_Complete.pth'):
    checkpoint = torch.load('MTL_ViT_Complete.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
else:
    print("WARNING: No model weights found. Predictions will be random.")

model = model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return {}, {}, ""

    img = Image.fromarray(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        disease_logits, nutrient_logits = model(input_tensor)

    disease_probs = F.softmax(disease_logits, dim=1)[0]
    disease_pred = disease_probs.argmax().item()
    disease_conf = disease_probs[disease_pred].item()
    disease_results = {DISEASE_NAMES[i]: float(disease_probs[i]) for i in range(len(DISEASE_NAMES))}

    nutrient_probs = F.softmax(nutrient_logits, dim=1)[0]
    nutrient_pred = nutrient_probs.argmax().item()
    nutrient_conf = nutrient_probs[nutrient_pred].item()
    nutrient_results = {NUTRIENT_NAMES[i]: float(nutrient_probs[i]) for i in range(len(NUTRIENT_NAMES))}

    is_healthy = (disease_pred == 2)
    healthy_conf = disease_probs[2].item()
    nutrient_uncertain = (nutrient_conf < 0.70)

    if is_healthy and healthy_conf > 0.90 and nutrient_uncertain:
        nutrient_results = {"No Deficiency Detected (Healthy Leaf)": 1.0}
        summary = f"HEALTHY LEAF\n\nDisease: None detected ({healthy_conf*100:.1f}% healthy)\nNutrient: No deficiency detected"
    elif is_healthy and nutrient_conf >= 0.70:
        summary = f"NUTRIENT DEFICIENCY DETECTED\n\nDisease: None ({healthy_conf*100:.1f}% healthy)\nNutrient: {NUTRIENT_NAMES[nutrient_pred]} ({nutrient_conf*100:.1f}%)"
    elif not is_healthy and disease_conf > 0.70:
        if nutrient_conf >= 0.70:
            summary = f"DISEASE + NUTRIENT DEFICIENCY\n\nDisease: {DISEASE_NAMES[disease_pred]} ({disease_conf*100:.1f}%)\nNutrient: {NUTRIENT_NAMES[nutrient_pred]} ({nutrient_conf*100:.1f}%)"
        else:
            nutrient_results = {"No Clear Deficiency Signal": 1.0}
            summary = f"DISEASE DETECTED\n\nDisease: {DISEASE_NAMES[disease_pred]} ({disease_conf*100:.1f}%)\nNutrient: No clear deficiency detected"
    else:
        summary = f"UNCERTAIN\n\nDisease: {DISEASE_NAMES[disease_pred]} ({disease_conf*100:.1f}%)\nNutrient: {NUTRIENT_NAMES[nutrient_pred]} ({nutrient_conf*100:.1f}%)\n\nPlease upload a clearer image."

    return disease_results, nutrient_results, summary

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Rice Leaf Image"),
    outputs=[
        gr.Label(num_top_classes=6, label="Disease Classification"),
        gr.Label(num_top_classes=3, label="Nutrient Deficiency Detection"),
        gr.Textbox(label="Diagnosis Summary", lines=5)
    ],
    title="MTL-ViT: Rice Leaf Health Analyzer",
    description="Upload a rice leaf image to simultaneously detect diseases (6 classes) and nutrient deficiencies (3 classes) using a Multi-Task Vision Transformer.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
