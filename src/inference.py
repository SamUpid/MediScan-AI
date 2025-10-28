"""
inference.py - MediScan-AI Inference Pipeline
Standalone module for pneumonia detection from chest X-rays
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import onnxruntime as ort
import os

class MediScanPredictor:
    """
    Complete inference pipeline for MediScan-AI
    Handles preprocessing, prediction for pneumonia detection
    """
    
    def __init__(self, model_path=None, onnx_path=None):
        """
        Initialize predictor with either PyTorch or ONNX model
        
        Args:
            model_path: Path to PyTorch model (.pth)
            onnx_path: Path to ONNX model (.onnx)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model (prefer ONNX for production)
        if onnx_path and os.path.exists(onnx_path):
            self.model_type = 'onnx'
            try:
                self.model = ort.InferenceSession(onnx_path)
                self.input_name = self.model.get_inputs()[0].name
                self.output_name = self.model.get_outputs()[0].name
                print(f"Loaded ONNX model from {onnx_path}")
            except Exception as e:
                raise ValueError(f"Failed to load ONNX model: {e}")
        elif model_path and os.path.exists(model_path):
            self.model_type = 'pytorch'
            self.model = self._build_model(num_classes=2, dropout_rate=0.5)
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"Loaded PyTorch model from {model_path}")
            except Exception as e:
                raise ValueError(f"Failed to load PyTorch model: {e}")
        else:
            raise ValueError("No valid model path provided")
        
        # Setup transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
        print(f"MediScanPredictor initialized with {self.model_type.upper()} model")
        print(f"Device: {self.device}")
    
    def _build_model(self, num_classes=2, dropout_rate=0.5):
        """Build ResNet50 model architecture"""
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
    
    def preprocess(self, image_path):
        """Load and preprocess image from file path"""
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transforms(image)
            return tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            raise ValueError(f"Error preprocessing image {image_path}: {e}")
    
    def predict(self, image_tensor):
        """Get prediction from image tensor"""
        try:
            if self.model_type == 'pytorch':
                with torch.no_grad():
                    image_tensor = image_tensor.to(self.device)
                    output = self.model(image_tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                return predicted.item(), confidence.item()
            else:  # ONNX
                input_data = image_tensor.numpy().astype(np.float32)
                outputs = self.model.run([self.output_name], {self.input_name: input_data})
                probs = torch.softmax(torch.from_numpy(outputs[0]), dim=1)
                confidence, predicted = torch.max(probs, 1)
                return predicted.item(), confidence.item()
        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")
    
    def predict_full(self, image_path):
        """
        Complete prediction pipeline
        
        Returns:
            dict with prediction results and metadata
        """
        try:
            # Preprocess
            image_tensor = self.preprocess(image_path)
            
            # Predict
            pred_class, confidence = self.predict(image_tensor)
            class_name = self.class_names[pred_class]
            
            # Get recommendation
            recommendation = self._get_recommendation(confidence)
            
            return {
                'class': class_name,
                'class_index': pred_class,
                'confidence': confidence,
                'recommendation': recommendation,
                'model_type': self.model_type,
                'success': True,
                'image_path': image_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'class': 'ERROR',
                'confidence': 0.0,
                'recommendation': 'Processing failed - please check image file'
            }
    
    def _get_recommendation(self, confidence):
        """Provide clinical recommendation based on confidence"""
        try:
            if confidence > 0.95:
                return "High confidence prediction"
            elif confidence > 0.85:
                return "Moderate confidence - Consider additional review"
            elif confidence > 0.70:
                return "Low confidence - Expert review recommended"
            else:
                return "Very low confidence - Requires expert validation"
        except Exception as e:
            return f"Error generating recommendation: {e}"
        
    def generate_gradcam(self, image_path, target_class=None):
        """Generate Grad-CAM visualization (PyTorch only)"""
        if self.model_type != 'pytorch':
            print("Grad-CAM only available for PyTorch models")
            return None
        
        try:
            import cv2
            import numpy as np
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            
            print(f"Attempting Grad-CAM for image: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = np.array(image)
            
            # Resize for visualization
            original_image_resized = cv2.resize(original_image, (224, 224))
            original_image_normalized = original_image_resized.astype(np.float32) / 255.0
            
            print(f"Model type: {type(self.model)}")
            
            # FIXED: Enable gradients for the target layer
            # Temporarily enable gradients for Grad-CAM
            for param in self.model.parameters():
                param.requires_grad = True
            
            # FIXED: Use correct target layer for ResNet50
            target_layer = None
            
            # Option 1: Try layer4 (the last convolutional layer in ResNet50)
            if hasattr(self.model, 'layer4'):
                target_layer = [self.model.layer4[-1]]
                print("Using layer4[-1] as target layer")
            else:
                # Find the last convolutional layer
                for name, module in reversed(list(self.model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = [module]
                        print(f"Using {name} as target layer")
                        break
            
            if target_layer is None:
                print("ERROR: Could not find suitable target layer for Grad-CAM")
                return None
            
            # FIXED: Remove use_cuda parameter - use device directly
            cam = GradCAM(
                model=self.model, 
                target_layers=target_layer
            )
            
            # Preprocess for model
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Get prediction if target_class not specified
            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor)
                    predicted = torch.argmax(output, 1).item()
                target_class = predicted
                print(f"Predicted class for Grad-CAM: {target_class} ({self.class_names[target_class]})")
            
            # Generate Grad-CAM
            targets = [ClassifierOutputTarget(target_class)]
            
            print("Generating Grad-CAM heatmap...")
            
            # Generate the heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            print(f"Grad-CAM shape: {grayscale_cam.shape}")
            print(f"Image shape: {original_image_normalized.shape}")
            
            # Overlay heatmap on original image
            visualization = show_cam_on_image(
                original_image_normalized,
                grayscale_cam,
                use_rgb=True
            )
            
            # FIXED: Convert back to disable gradients after Grad-CAM
            for param in self.model.parameters():
                param.requires_grad = False
            
            print("✅ Grad-CAM generated successfully!")
            
            # Convert BGR to RGB for display
            visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            return visualization
                
        except ImportError as e:
            print(f"Grad-CAM dependencies not available: {e}")
            return None
        except Exception as e:
            print(f"❌ Error generating Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Example usage"""
    try:
        predictor = MediScanPredictor(model_path='notebooks/models/resnet50_fold_2_best.pth')
        print("MediScanPredictor ready!")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")

if __name__ == "__main__":
    main()