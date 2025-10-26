# MediScan-AI Model Card

## Model Details
- **Model Name**: MediScan-AI Pneumonia Detector
- **Model Type**: ResNet50 with custom classifier head
- **Framework**: PyTorch 2.8.0+cu126
- **Training Date**: October 2024
- **Version**: 1.0
- **Architecture**: Transfer learning from ImageNet-pretrained ResNet50

## Performance Summary
- **Clean Image Accuracy**: 100.0%
- **Average Perturbed Accuracy**: 10.0%
- **Robustness Gap**: 100.0%
- **Average Prediction Confidence**: 87.1%

## Usage
- Use as preliminary screening tool only
- Always validate with expert radiologist
- Be aware of sensitivity to image quality issues

*Last Updated: 2025-10-26*
