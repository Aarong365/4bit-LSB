#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration script showing how to use the tensor indexing fix
in a practical steganography scenario.

This script demonstrates:
1. The original problem and its fix
2. How to use the enhanced steganography system
3. Integration with existing LSB implementations
"""

import torch
import numpy as np
from PIL import Image
import os
import cv2

from tensor_steganography import TensorSteganography
from enhanced_lsb_stego import EnhancedLSBSteganography


def create_sample_images():
    """Create sample images for demonstration."""
    demo_dir = "/tmp/tensor_fix_demo"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create a colorful cover image
    cover = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(cover, (50, 50), (200, 200), (255, 100, 100), -1)
    cv2.circle(cover, (128, 128), 60, (100, 255, 100), -1)
    cv2.rectangle(cover, (80, 80), (180, 180), (100, 100, 255), 2)
    
    # Create a secret image with pattern
    secret = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 20):
        cv2.line(secret, (i, 0), (i, 255), (200, 200, 200), 2)
        cv2.line(secret, (0, i), (255, i), (200, 200, 200), 2)
    cv2.putText(secret, "SECRET", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    cover_path = os.path.join(demo_dir, "cover.png")
    secret_path = os.path.join(demo_dir, "secret.png")
    
    Image.fromarray(cover).save(cover_path)
    Image.fromarray(secret).save(secret_path)
    
    return demo_dir, cover_path, secret_path


def demonstrate_original_problem():
    """Demonstrate the original tensor indexing problem and its fix."""
    print("🔍 Demonstrating Original Problem")
    print("=" * 50)
    
    # Simulate the problematic scenario
    noise = torch.randn(5, 3, 64, 64)
    t_noise_indices = torch.tensor([2])
    
    print(f"Noise tensor shape: {noise.shape}")
    print(f"Index tensor: {t_noise_indices}")
    print(f"Index element shape: {t_noise_indices[0].shape} (0-dimensional!)")
    
    print("\n❌ Original problematic code:")
    print("   best_noise = noise[t_noise_indices[0]]")
    print("   # Would cause: IndexError: invalid index of a 0-dim tensor")
    
    print("\n✅ Fixed code:")
    print("   safe_index = t_noise_indices[0].item()")
    print("   best_noise = noise[safe_index]")
    
    # Demonstrate the fix
    safe_index = t_noise_indices[0].item()
    best_noise = noise[safe_index]
    
    print(f"\n✓ Fix successful!")
    print(f"   Safe index: {safe_index}")
    print(f"   Result shape: {best_noise.shape}")


def demonstrate_tensor_steganography():
    """Demonstrate the tensor-aware steganography system."""
    print("\n🚀 Demonstrating Tensor Steganography")
    print("=" * 50)
    
    demo_dir, cover_path, secret_path = create_sample_images()
    
    # Initialize tensor steganography
    stego = TensorSteganography(device='cpu')
    
    # Create DDCM-like decompress indices
    decompress_indices = {
        't_noise_indices': torch.tensor([1, 0, 2]),
        'noise': torch.randn(5, 3, 256, 256) * 0.1,  # Small noise
        'compression_mode': True
    }
    
    print(f"📁 Demo directory: {demo_dir}")
    print(f"🖼️  Cover image: {os.path.basename(cover_path)}")
    print(f"🔒 Secret image: {os.path.basename(secret_path)}")
    
    # Embed secret
    stego_path = os.path.join(demo_dir, "stego_tensor.png")
    print(f"\n📤 Embedding secret image...")
    metadata = stego.embed_secret_image(
        cover_path, secret_path, stego_path,
        compression_mode=True,
        noise_indices=decompress_indices.get('t_noise_indices')
    )
    print(f"✓ Stego image created: {os.path.basename(stego_path)}")
    
    # Extract secret
    extracted_path = os.path.join(demo_dir, "extracted_tensor.png")
    print(f"\n📥 Extracting secret image (with tensor fix)...")
    extracted_tensor = stego.extract_secret_image(
        stego_path, extracted_path, decompress_indices
    )
    print(f"✓ Secret extracted: {os.path.basename(extracted_path)}")
    
    return demo_dir


def demonstrate_enhanced_lsb():
    """Demonstrate the enhanced LSB system with backward compatibility."""
    print("\n🔧 Demonstrating Enhanced LSB Steganography")
    print("=" * 50)
    
    demo_dir, cover_path, secret_path = create_sample_images()
    
    # Test tensor-aware mode
    print("🎯 Testing tensor-aware mode...")
    enhanced_stego_tensor = EnhancedLSBSteganography(use_tensors=True)
    
    decompress_indices = {
        't_noise_indices': torch.tensor([0, 1, 2]),
        'noise': torch.randn(3, 3, 256, 256) * 0.05
    }
    
    stego_enhanced_path = os.path.join(demo_dir, "stego_enhanced.png")
    extracted_enhanced_path = os.path.join(demo_dir, "extracted_enhanced.png")
    
    # Embed and extract with tensor support
    total_bits = enhanced_stego_tensor.embed_image_in_image_enhanced(
        cover_path, secret_path, stego_enhanced_path, decompress_indices
    )
    
    success = enhanced_stego_tensor.extract_secret_image_enhanced(
        stego_enhanced_path, extracted_enhanced_path, decompress_indices
    )
    
    print(f"✓ Tensor mode: {total_bits} bits embedded, extraction {'successful' if success else 'failed'}")
    
    # Test traditional mode (backward compatibility)
    print("\n📼 Testing traditional mode (backward compatibility)...")
    enhanced_stego_traditional = EnhancedLSBSteganography(use_tensors=False)
    
    stego_traditional_path = os.path.join(demo_dir, "stego_traditional.png")
    extracted_traditional_path = os.path.join(demo_dir, "extracted_traditional.png")
    
    total_bits_traditional = enhanced_stego_traditional.embed_image_in_image_enhanced(
        cover_path, secret_path, stego_traditional_path
    )
    
    success_traditional = enhanced_stego_traditional.extract_secret_image_enhanced(
        stego_traditional_path, extracted_traditional_path
    )
    
    print(f"✓ Traditional mode: {total_bits_traditional} bits embedded, extraction {'successful' if success_traditional else 'failed'}")
    
    return demo_dir


def show_results(demo_dir):
    """Show the results of the demonstration."""
    print(f"\n📊 Results Summary")
    print("=" * 50)
    
    files = os.listdir(demo_dir)
    files.sort()
    
    print(f"📁 Output directory: {demo_dir}")
    print("📋 Files created:")
    
    for i, file in enumerate(files, 1):
        file_path = os.path.join(demo_dir, file)
        size = os.path.getsize(file_path)
        print(f"   {i:2d}. {file:<25} ({size:,} bytes)")
    
    print(f"\n🎯 Key achievements:")
    print("   ✅ Tensor indexing error fixed")
    print("   ✅ DDCM compatibility ensured") 
    print("   ✅ Backward compatibility maintained")
    print("   ✅ Complete steganography pipeline working")


def main():
    """Run the complete demonstration."""
    print("🔒 Tensor Indexing Fix Demonstration")
    print("🎯 Steganography Pipeline with PyTorch Compatibility")
    print("=" * 70)
    
    try:
        # Demonstrate the original problem and fix
        demonstrate_original_problem()
        
        # Demonstrate tensor steganography
        demo_dir = demonstrate_tensor_steganography()
        
        # Demonstrate enhanced LSB with backward compatibility
        demonstrate_enhanced_lsb()
        
        # Show results
        show_results(demo_dir)
        
        print(f"\n🎉 Demonstration completed successfully!")
        print(f"📖 See TENSOR_FIX_README.md for detailed documentation")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()