"""
Inference script for making predictions on new chest X-ray images
Usage: python predict.py --image path/to/xray.jpg
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import cv2
import os

# Configuration
MODEL_PATH = "mobilenet_cxr.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

class GrayToRGB(object):
    """Convert grayscale image to RGB by duplicating channels"""
    def __call__(self, img):
        arr = np.array(img)
        if len(arr.shape) == 2:
            arr = np.stack([arr]*3, axis=-1)
        return Image.fromarray(arr)

# Image preprocessing pipeline
transform = transforms.Compose([
    GrayToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
])

def load_model(model_path, num_classes=2):
    """Load the trained MobileNetV2 model"""
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, image_path, show_gradcam=False):
    """
    Make prediction on a single chest X-ray image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the X-ray image
        show_gradcam: Whether to display Grad-CAM visualization
    
    Returns:
        predicted_class, confidence, probabilities
    """
    # Load and preprocess image
    img_pil = Image.open(image_path).convert('L')  # Convert to grayscale
    img_tensor: torch.Tensor = transform(img_pil).unsqueeze(0).to(DEVICE)  # type: ignore
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    predicted_idx = int(predicted.item())
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence_value = confidence.item()
    probs_dict = {CLASS_NAMES[i]: probs[0][i].item() for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence_value, probs_dict

def visualize_prediction(image_path, predicted_class, confidence, probs_dict):
    """Visualize the prediction with the X-ray image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot X-ray image
    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title(f'Chest X-Ray\n{os.path.basename(image_path)}', fontsize=12, fontweight='bold')
    
    # Prediction box with color coding
    color = 'green' if predicted_class == 'NORMAL' else 'red'
    axes[0].text(0.5, -0.1, f'Prediction: {predicted_class}',
                transform=axes[0].transAxes, ha='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                fontweight='bold')
    
    # Plot probability bar chart
    classes = list(probs_dict.keys())
    probabilities = list(probs_dict.values())
    colors_bar = ['green' if c == 'NORMAL' else 'red' for c in classes]
    
    bars = axes[1].barh(classes, probabilities, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Confidence', fontsize=12, fontweight='bold')
    axes[1].set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        axes[1].text(prob + 0.02, i, f'{prob*100:.1f}%', 
                    va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'prediction_{os.path.basename(image_path)}', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def batch_predict(model, image_dir, output_csv='predictions.csv'):
    """Make predictions on all images in a directory"""
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        print("⚠️  Warning: pandas not installed. Install with: pip install pandas")
        print("Proceeding without CSV export...")
        pd = None  # type: ignore
    
    results = []
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n🔍 Processing {len(image_files)} images from {image_dir}...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            pred_class, confidence, probs = predict_image(model, img_path)
            results.append({
                'filename': img_file,
                'prediction': pred_class,
                'confidence': f'{confidence:.4f}',
                'normal_prob': f'{probs["NORMAL"]:.4f}',
                'pneumonia_prob': f'{probs["PNEUMONIA"]:.4f}'
            })
            print(f"✓ {img_file}: {pred_class} ({confidence*100:.1f}%)")
        except Exception as e:
            print(f"✗ {img_file}: Error - {str(e)}")
    
    # Save results to CSV if pandas is available
    if pd is not None:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to {output_csv}")
        return df
    else:
        # Print results to console if pandas not available
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        for result in results:
            print(f"{result['filename']:30s} | {result['prediction']:10s} | {result['confidence']}")
        print("="*70)
        return results

def main():
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-ray images')
    parser.add_argument('--image', type=str, help='Path to a single X-ray image')
    parser.add_argument('--dir', type=str, help='Directory containing multiple X-ray images')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model weights')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file '{args.model}' not found!")
        print("Please train the model first using train_cxray.py")
        return
    
    # Load model
    print("🔧 Loading model...")
    model = load_model(args.model)
    print(f"✓ Model loaded from {args.model}")
    print(f"✓ Using device: {DEVICE}")
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Error: Image file '{args.image}' not found!")
            return
        
        print(f"\n🔍 Analyzing: {args.image}")
        predicted_class, confidence, probs_dict = predict_image(model, args.image)
        
        # Print results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("\nClass Probabilities:")
        for cls, prob in probs_dict.items():
            print(f"  {cls:12s}: {prob*100:.2f}%")
        print("="*60)
        
        # Visualize
        if not args.no_viz:
            visualize_prediction(args.image, predicted_class, confidence, probs_dict)
    
    # Batch prediction
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"❌ Error: Directory '{args.dir}' not found!")
            return
        
        result = batch_predict(model, args.dir)
        if result is not None:
            try:
                import pandas as pd  # type: ignore
                if isinstance(result, pd.DataFrame):
                    print(f"\n📊 Summary:")
                    print(result['prediction'].value_counts())  # type: ignore
            except (ImportError, AttributeError):
                pass
    
    else:
        print("❌ Please specify either --image or --dir")
        parser.print_help()

if __name__ == "__main__":
    main()
