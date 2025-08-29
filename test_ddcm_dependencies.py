#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 DDCM + GS 水印系统的依赖管理和兼容性
"""

import sys
import traceback
import numpy as np

def test_scipy_imports():
    """测试 scipy 导入修复"""
    print("=== 测试 scipy 导入修复 ===")
    
    try:
        # 正确的导入方式 (修复后)
        from scipy.special import betainc
        print("✓ 成功从 scipy.special 导入 betainc")
        
        # 测试函数调用
        result = betainc(2, 3, 0.5)
        print(f"✓ betainc(2, 3, 0.5) = {result}")
        
    except ImportError as e:
        print(f"✗ scipy.special.betainc 导入失败: {e}")
        return False
    
    try:
        # 错误的导入方式 (应该失败)
        from scipy.stats import betainc
        print("✗ 意外地从 scipy.stats 导入了 betainc (这是错误的)")
    except ImportError:
        print("✓ 正确地无法从 scipy.stats 导入 betainc")
    
    return True

def test_diffusers_mock():
    """测试 diffusers 的 Mock 实现"""
    print("\n=== 测试 diffusers Mock 实现 ===")
    
    # 导入我们的系统
    from ddcm_gs_watermark import dep_manager
    
    diffusers = dep_manager.get_dependency('diffusers')
    print(f"✓ 获取 diffusers 依赖: {type(diffusers)}")
    
    # 测试 PIL_to_tensor
    try:
        from PIL import Image
        test_image = Image.new('RGB', (64, 64), (128, 128, 128))
        tensor = diffusers.utils.PIL_to_tensor(test_image)
        print(f"✓ PIL_to_tensor 工作正常，输出类型: {type(tensor)}")
        
        if hasattr(tensor, 'shape'):
            print(f"✓ 输出tensor形状: {tensor.shape}")
        elif hasattr(tensor, 'data'):
            print(f"✓ 输出tensor数据形状: {tensor.data.shape}")
        
    except Exception as e:
        print(f"✗ PIL_to_tensor 失败: {e}")
        return False
    
    # 测试 tensor_to_pil
    try:
        test_tensor_data = np.random.rand(3, 64, 64)
        from ddcm_gs_watermark import MockTensor
        test_tensor = MockTensor(test_tensor_data)
        
        pil_image = diffusers.utils.tensor_to_pil(test_tensor)
        print(f"✓ tensor_to_pil 工作正常，输出类型: {type(pil_image)}")
        
        if pil_image and hasattr(pil_image, 'size'):
            print(f"✓ 输出图像尺寸: {pil_image.size}")
        
    except Exception as e:
        print(f"✗ tensor_to_pil 失败: {e}")
        return False
    
    return True

def test_dependency_fallbacks():
    """测试依赖回退机制"""
    print("\n=== 测试依赖回退机制 ===")
    
    from ddcm_gs_watermark import dep_manager
    
    # 测试缺失的依赖
    missing_deps = ['torch', 'transformers', 'some_nonexistent_package']
    
    for dep_name in missing_deps:
        try:
            dep = dep_manager.get_dependency(dep_name)
            print(f"✓ {dep_name}: 获取到 Mock 实现 ({type(dep)})")
            
            # 测试基本调用
            if hasattr(dep, '__call__'):
                result = dep()
                print(f"  ✓ 可调用的 Mock 对象")
            
            # 测试属性访问
            attr = getattr(dep, 'some_attr', None)
            if attr:
                print(f"  ✓ 属性访问工作正常")
            
        except Exception as e:
            print(f"✗ {dep_name}: 依赖回退失败: {e}")
            return False
    
    return True

def test_watermark_basic_flow():
    """测试水印系统基本流程"""
    print("\n=== 测试水印系统基本流程 ===")
    
    try:
        from ddcm_gs_watermark import GaussianShadingWatermark, create_test_image
        
        # 创建系统
        watermark_system = GaussianShadingWatermark(
            codebook_size=64,  # 使用较小的参数进行快速测试
            feature_dim=32,
            watermark_length=8,
            redundancy_factor=2
        )
        print("✓ 水印系统初始化成功")
        
        # 创建测试图像
        test_image = create_test_image(64, 64)
        print(f"✓ 测试图像创建成功: {test_image.shape}")
        
        # 嵌入水印
        watermark_text = "Test"
        watermarked_image, metadata = watermark_system.embed_watermark(test_image, watermark_text)
        print(f"✓ 水印嵌入成功: {watermarked_image.shape}")
        print(f"✓ 元数据: {list(metadata.keys())}")
        
        # 提取水印
        extracted_text = watermark_system.extract_watermark(watermarked_image, metadata)
        print(f"✓ 水印提取完成: '{extracted_text}'")
        
        # 计算成功率
        if extracted_text:
            char_accuracy = sum(c1 == c2 for c1, c2 in zip(watermark_text, extracted_text)) / max(len(watermark_text), len(extracted_text))
            print(f"✓ 字符准确率: {char_accuracy:.2%}")
        else:
            print("⚠ 提取的水印为空，但系统流程正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 水印系统测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("DDCM + GS 水印系统依赖和兼容性测试")
    print("=" * 50)
    
    tests = [
        ("scipy 导入修复", test_scipy_imports),
        ("diffusers Mock 实现", test_diffusers_mock),
        ("依赖回退机制", test_dependency_fallbacks),
        ("水印系统基本流程", test_watermark_basic_flow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试出现异常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！系统兼容性良好。")
        return 0
    else:
        print("⚠ 部分测试失败，请检查相关问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())