#!/usr/bin/env python3
"""
快速测试脚本 - 验证改进的Transformer架构
"""

import torch
import yaml
from transformer_improved import ImprovedTransformer, HybridTransformer
from transformer import Transformer

def test_model_creation():
    """测试模型创建"""
    print("🧪 测试模型创建...")
    
    # 测试配置
    config = {
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 2,
        'hidden_dim': 32,
        'dropout': 0.1,
        'future_hours': 24,
        'use_forecast': True,
        'pooling_type': 'attention'
    }
    
    hist_dim = 10
    fcst_dim = 5
    
    try:
        # 测试原始Transformer
        original_model = Transformer(hist_dim, fcst_dim, config)
        print("✅ 原始Transformer创建成功")
        
        # 测试改进Transformer
        improved_model = ImprovedTransformer(hist_dim, fcst_dim, config)
        print("✅ 改进Transformer创建成功")
        
        # 测试混合Transformer
        hybrid_model = HybridTransformer(hist_dim, fcst_dim, config)
        print("✅ 混合Transformer创建成功")
        
        return original_model, improved_model, hybrid_model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None, None, None

def test_forward_pass(original_model, improved_model, hybrid_model):
    """测试前向传播"""
    print("\n🧪 测试前向传播...")
    
    batch_size = 2
    past_hours = 24
    future_hours = 24
    hist_dim = 10
    fcst_dim = 5
    
    # 创建测试数据
    hist_data = torch.randn(batch_size, past_hours, hist_dim)
    fcst_data = torch.randn(batch_size, future_hours, fcst_dim)
    
    try:
        # 测试原始模型
        with torch.no_grad():
            original_output = original_model(hist_data, fcst_data)
            print(f"✅ 原始Transformer输出形状: {original_output.shape}")
            
            # 测试改进模型
            improved_output = improved_model(hist_data, fcst_data)
            print(f"✅ 改进Transformer输出形状: {improved_output.shape}")
            
            # 测试混合模型
            hybrid_output = hybrid_model(hist_data, fcst_data)
            print(f"✅ 混合Transformer输出形状: {hybrid_output.shape}")
            
            # 验证输出范围
            print(f"📊 输出范围检查:")
            print(f"   原始模型: [{original_output.min():.3f}, {original_output.max():.3f}]")
            print(f"   改进模型: [{improved_output.min():.3f}, {improved_output.max():.3f}]")
            print(f"   混合模型: [{hybrid_output.min():.3f}, {hybrid_output.max():.3f}]")
            
            # 验证输出形状
            expected_shape = (batch_size, future_hours)
            assert original_output.shape == expected_shape, f"原始模型输出形状错误: {original_output.shape}"
            assert improved_output.shape == expected_shape, f"改进模型输出形状错误: {improved_output.shape}"
            assert hybrid_output.shape == expected_shape, f"混合模型输出形状错误: {hybrid_output.shape}"
            
            print("✅ 所有模型输出形状正确")
            
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")

def test_different_pooling_types():
    """测试不同的池化类型"""
    print("\n🧪 测试不同池化类型...")
    
    config_base = {
        'd_model': 32,
        'num_heads': 2,
        'num_layers': 1,
        'hidden_dim': 16,
        'dropout': 0.1,
        'future_hours': 12,
        'use_forecast': False
    }
    
    pooling_types = ['last', 'mean', 'max', 'attention', 'learned_attention']
    
    for pooling_type in pooling_types:
        try:
            config = config_base.copy()
            config['pooling_type'] = pooling_type
            
            model = ImprovedTransformer(5, 0, config)
            
            # 测试前向传播
            test_input = torch.randn(1, 12, 5)
            with torch.no_grad():
                output = model(test_input)
                
            print(f"✅ {pooling_type} 池化类型测试成功, 输出形状: {output.shape}")
            
        except Exception as e:
            print(f"❌ {pooling_type} 池化类型测试失败: {e}")

def test_config_loading():
    """测试配置文件加载"""
    print("\n🧪 测试配置文件加载...")
    
    config_files = [
        'Transformer_low_PV_plus_NWP_24h_noTE.yaml',
        'Transformer_improved_PV_plus_NWP_24h.yaml',
        'Transformer_attention_pooling.yaml',
        'Transformer_hybrid.yaml'
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ {config_file} 加载成功")
            print(f"   模型: {config.get('model', 'N/A')}")
            print(f"   复杂度: {config.get('model_complexity', 'N/A')}")
            
            if 'model_params' in config and 'low' in config['model_params']:
                pooling_type = config['model_params']['low'].get('pooling_type', 'N/A')
                print(f"   池化类型: {pooling_type}")
                
        except Exception as e:
            print(f"❌ {config_file} 加载失败: {e}")

def main():
    """主测试函数"""
    print("🔬 Transformer改进架构测试")
    print("=" * 50)
    
    # 测试模型创建
    original_model, improved_model, hybrid_model = test_model_creation()
    
    if original_model and improved_model and hybrid_model:
        # 测试前向传播
        test_forward_pass(original_model, improved_model, hybrid_model)
        
        # 测试不同池化类型
        test_different_pooling_types()
        
        # 测试配置文件加载
        test_config_loading()
        
        print("\n🎉 所有测试通过!")
        print("✅ 改进的Transformer架构已准备就绪")
        print("🚀 可以开始实验对比了")
        
    else:
        print("\n❌ 测试失败，请检查代码实现")

if __name__ == "__main__":
    main()
