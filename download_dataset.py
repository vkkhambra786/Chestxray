"""
Automatic Dataset Downloader for Chest X-Ray Project
This script downloads the dataset from Kaggle and sets it up properly
"""

import os
import shutil
import kagglehub

print("="*70)
print("📥 CHEST X-RAY DATASET DOWNLOADER")
print("="*70)
print("\nThis will download ~1.15 GB of data")
print("Please be patient, this may take 5-15 minutes depending on your connection\n")

try:
    # Download latest version from Kaggle
    print("🔽 Downloading dataset from Kaggle...")
    print("   (First time may require Kaggle authentication)\n")
    
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    print(f"\n✓ Download complete!")
    print(f"📁 Downloaded to: {path}")
    
    # Check if dataset was downloaded
    if not os.path.exists(path):
        print("❌ Error: Download path doesn't exist")
        exit(1)
    
    # Find the chest_xray folder in the downloaded path
    chest_xray_source = None
    
    # kagglehub might download to different structures, let's find it
    if os.path.exists(os.path.join(path, "chest_xray")):
        chest_xray_source = os.path.join(path, "chest_xray")
    elif os.path.exists(os.path.join(path, "chest-xray-pneumonia", "chest_xray")):
        chest_xray_source = os.path.join(path, "chest-xray-pneumonia", "chest_xray")
    else:
        # Search for it
        for root, dirs, files in os.walk(path):
            if "chest_xray" in dirs:
                chest_xray_source = os.path.join(root, "chest_xray")
                break
    
    if not chest_xray_source:
        print(f"❌ Error: Could not find 'chest_xray' folder in downloaded data")
        print(f"   Please check: {path}")
        print(f"\n   You can manually copy the 'chest_xray' folder to:")
        print(f"   {os.getcwd()}")
        exit(1)
    
    print(f"\n✓ Found dataset at: {chest_xray_source}")
    
    # Create symbolic link or copy to project directory
    target_path = os.path.join(os.getcwd(), "chest_xray")
    
    if os.path.exists(target_path):
        print(f"\n⚠️  'chest_xray' folder already exists in project directory")
        response = input("   Do you want to replace it? (y/n): ").lower()
        if response != 'y':
            print("   Keeping existing dataset")
        else:
            print("   Removing old dataset...")
            if os.path.islink(target_path):
                os.unlink(target_path)
            else:
                shutil.rmtree(target_path)
            print("   Creating link to new dataset...")
            os.symlink(chest_xray_source, target_path, target_is_directory=True)
            print(f"   ✓ Linked to: {chest_xray_source}")
    else:
        print(f"\n🔗 Creating link to dataset in project directory...")
        try:
            # Try to create symbolic link (faster, saves space)
            os.symlink(chest_xray_source, target_path, target_is_directory=True)
            print(f"   ✓ Symbolic link created: chest_xray -> {chest_xray_source}")
        except OSError:
            # If symlink fails (e.g., no admin rights), copy instead
            print("   Symbolic link failed, copying files instead (this may take a few minutes)...")
            shutil.copytree(chest_xray_source, target_path)
            print(f"   ✓ Dataset copied to: {target_path}")
    
    # Verify dataset structure
    print("\n📊 Verifying dataset structure...")
    
    required_dirs = [
        'train/NORMAL',
        'train/PNEUMONIA',
        'val/NORMAL',
        'val/PNEUMONIA',
        'test/NORMAL',
        'test/PNEUMONIA'
    ]
    
    all_ok = True
    total_images = 0
    
    for subdir in required_dirs:
        full_path = os.path.join(target_path, subdir)
        if os.path.exists(full_path):
            count = len([f for f in os.listdir(full_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_images += count
            print(f"   ✓ {subdir:25s}: {count:4d} images")
        else:
            print(f"   ✗ {subdir:25s}: NOT FOUND")
            all_ok = False
    
    if all_ok:
        print(f"\n✅ SUCCESS! Dataset is ready!")
        print(f"   Total images: {total_images}")
        print(f"\n🚀 Next steps:")
        print(f"   1. Run: python check_setup.py  (verify everything)")
        print(f"   2. Run: python train_cxray.py  (start training)")
    else:
        print(f"\n⚠️  Dataset structure incomplete")
        print(f"   Some folders are missing. Please check the dataset.")
    
    print("\n" + "="*70)

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print(f"\n💡 Alternative options:")
    print(f"   1. Download manually from:")
    print(f"      https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print(f"   2. See DATASET_GUIDE.md for detailed instructions")
    print(f"\n   If you need Kaggle authentication:")
    print(f"   - Go to https://www.kaggle.com/settings/account")
    print(f"   - Create API token (downloads kaggle.json)")
    print(f"   - Place kaggle.json in: %USERPROFILE%\\.kaggle\\")
    import traceback
    traceback.print_exc()
