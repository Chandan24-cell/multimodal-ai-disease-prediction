#!/usr/bin/env python3
"""
Verification script to test:
1. ViT model loads successfully
2. Outputs 768-dimensional embeddings
3. Model.vit attribute exists for Grad-CAM
"""
import sys
import torch
from PIL import Image
import io

print("=" * 60)
print("ViT Model Verification Script")
print("=" * 60)

# Test 1: Import model
print("\n[1/5] Importing MedicalViTModel...")
try:
    from models.vit_model import MedicalViTModel
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize model
print("\n[2/5] Initializing MedicalViTModel...")
try:
    model = MedicalViTModel(num_labels=14)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✓ Model initialized on device: {device}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Check model.vit attribute exists
print("\n[3/5] Checking model.vit attribute (required for Grad-CAM)...")
try:
    assert hasattr(model, 'vit'), "Model missing 'vit' attribute"
    print(f"✓ model.vit exists: {type(model.vit)}")
except Exception as e:
    print(f"✗ Check failed: {e}")
    sys.exit(1)

# Test 4: Run inference with dummy image
print("\n[4/5] Running forward pass with dummy image...")
try:
    # Create dummy RGB image (224x224)
    dummy_img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    dummy_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Convert to tensor
    from torchvision import transforms
    processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    pixel_values = processor(dummy_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Test get_last_hidden_state
        last_hidden = model.get_last_hidden_state(pixel_values)
        print(f"  Last hidden state shape: {last_hidden.shape}")
        
        # Test forward pass
        outputs = model(pixel_values)
        logits = outputs.logits
        print(f"  Logits shape: {logits.shape}")
        
    print("✓ Forward pass successful")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify embedding dimension
print("\n[5/5] Verifying 768-dimensional embedding...")
try:
    with torch.no_grad():
        last_hidden = model.get_last_hidden_state(pixel_values)
        cls_token = last_hidden[:, 0, :].squeeze(0)  # (768,)
        embedding_dim = cls_token.shape[0]
        
        assert embedding_dim == 768, f"Expected 768-dim, got {embedding_dim}-dim"
        print(f"✓ CLS token embedding dimension: {embedding_dim}")
except Exception as e:
    print(f"✗ Embedding check failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL VERIFICATION TESTS PASSED")
print("=" * 60)
print("\nSummary:")
print("  • ViT model loads successfully")
print("  • Model outputs 768-dimensional embeddings")
print("  • model.vit attribute available for Grad-CAM")
print("  • Ready for production deployment")
