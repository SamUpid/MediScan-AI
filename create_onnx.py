# create_onnx.py
import torch
import torch.nn as nn
from torchvision import models
import os

print("🔄 Creating ONNX file from your trained model...")

def build_model(num_classes=2, dropout_rate=0.5):
    """Build the exact same model architecture you trained"""
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# Load your trained model
model = build_model()
checkpoint_path = 'notebooks/models/resnet50_fold_2_best.pth'

if os.path.exists(checkpoint_path):
    print(f"✓ Found your trained model: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded weights from model_state_dict")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded weights from checkpoint")
    
    model.eval()
    print("✓ Model prepared for conversion")
    
    # Export to ONNX
    onnx_path = 'models/mediscan_ai_best.onnx'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create dummy input (same as your training)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("🔄 Converting to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ ONNX model created: {onnx_path}")
    print(f"📊 File size: {os.path.getsize(onnx_path) / (1024*1024):.1f} MB")
    
    # Verify it was created
    if os.path.exists(onnx_path):
        print("🎉 Conversion successful! Your model is now in ONNX format.")
    else:
        print("❌ Conversion failed - file not created")
        
else:
    print(f"❌ Model file not found: {checkpoint_path}")
    print("Please check the file path")
