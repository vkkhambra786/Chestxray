"""
Quick verification script to check if your environment is ready for training
Run this BEFORE starting training to catch any issues early
"""

import os
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_mark(passed):
    """Return checkmark or X"""
    return "✓" if passed else "✗"

def main():
    print("\n🔍 CHEST X-RAY PROJECT - ENVIRONMENT CHECK")
    print("This script will verify your setup before training\n")
    
    all_passed = True
    
    # ========== CHECK 1: Python Version ==========
    print_header("1. Python Version")
    python_version = sys.version_info
    version_ok = python_version.major == 3 and python_version.minor >= 8
    print(f"{check_mark(version_ok)} Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    if not version_ok:
        print("   ⚠️  Warning: Python 3.8+ recommended")
        all_passed = False
    else:
        print("   ✓ Version OK")
    
    # ========== CHECK 2: Required Packages ==========
    print_header("2. Required Packages")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    for package, name in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✓ {name:20s} - Installed")
        except ImportError:
            print(f"✗ {name:20s} - MISSING")
            missing_packages.append(name)
            all_passed = False
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
    
    # ========== CHECK 3: CUDA/GPU ==========
    print_header("3. CUDA/GPU Support")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"{check_mark(cuda_available)} CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print("   ✓ Training will be FAST (GPU mode)")
        else:
            print("   ⚠️  No GPU detected - training will use CPU (slower)")
            print("   Expected training time: 2-4 hours")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
    
    # ========== CHECK 4: Dataset ==========
    print_header("4. Dataset Structure")
    
    dataset_path = "chest_xray"
    dataset_ok = True
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset folder '{dataset_path}' NOT FOUND")
        print(f"\n📥 NEXT STEPS:")
        print(f"   1. Download dataset from:")
        print(f"      https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print(f"   2. Extract the ZIP file to this directory")
        print(f"   3. Make sure 'chest_xray' folder exists here")
        print(f"\n   See DATASET_GUIDE.md for detailed instructions")
        all_passed = False
        dataset_ok = False
    else:
        print(f"✓ Dataset folder found: {dataset_path}")
        
        # Check subdirectories
        required_dirs = [
            'train/NORMAL',
            'train/PNEUMONIA',
            'val/NORMAL',
            'val/PNEUMONIA',
            'test/NORMAL',
            'test/PNEUMONIA'
        ]
        
        total_images = 0
        for subdir in required_dirs:
            full_path = os.path.join(dataset_path, subdir)
            if os.path.exists(full_path):
                count = len([f for f in os.listdir(full_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_images += count
                status = "✓" if count > 0 else "⚠️ "
                print(f"  {status} {subdir:25s}: {count:4d} images")
            else:
                print(f"  ✗ {subdir:25s}: NOT FOUND")
                all_passed = False
                dataset_ok = False
        
        if dataset_ok:
            print(f"\n  ✓ Total images found: {total_images}")
            if total_images < 5000:
                print(f"  ⚠️  Warning: Expected ~5,856 images, found {total_images}")
                print(f"     Dataset might be incomplete")
    
    # ========== CHECK 5: Disk Space ==========
    print_header("5. Disk Space")
    try:
        import shutil
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        print(f"✓ Free disk space: {free_gb:.2f} GB")
        if free_gb < 2:
            print(f"  ⚠️  Warning: Less than 2 GB free. May need more space.")
            print(f"     Dataset: ~2 GB, Model files: ~50 MB")
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
    
    # ========== CHECK 6: File Structure ==========
    print_header("6. Project Files")
    
    required_files = [
        'train_cxray.py',
        'predict.py',
        'visualize_results.py',
        'requirements.txt',
        'README.md'
    ]
    
    for filename in required_files:
        exists = os.path.exists(filename)
        print(f"{check_mark(exists)} {filename}")
        if not exists:
            print(f"   ⚠️  File missing - may affect functionality")
    
    # ========== FINAL SUMMARY ==========
    print_header("SUMMARY")
    
    if all_passed and dataset_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\n🚀 You're ready to start training!")
        print("\nRun this command to train:")
        print("   python train_cxray.py")
        print("\nExpected training time:")
        try:
            import torch
            if torch.cuda.is_available():
                print("   With GPU: 20-40 minutes")
            else:
                print("   With CPU: 2-4 hours")
        except:
            print("   With CPU: 2-4 hours")
    else:
        print("⚠️  SOME ISSUES FOUND")
        print("\nPlease fix the issues above before training.")
        
        if not dataset_ok:
            print("\n📥 PRIORITY: Download and extract the dataset")
            print("   See: DATASET_GUIDE.md or HOW_TO_RUN.md")
        
        if missing_packages:
            print("\n📦 Install missing packages:")
            print("   pip install -r requirements.txt")
    
    print("\n" + "="*70)
    print("\n💡 TIP: See HOW_TO_RUN.md for complete step-by-step guide")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Check interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during check: {e}")
        import traceback
        traceback.print_exc()
