#!/usr/bin/env python3
"""
简单测试注意力池化模型
"""

import yaml
import torch
import numpy as np
from transformer_improved import ImprovedTransformer

def main():
    print("🔍 简单测试注意力池化模型")
    print("=" * 50)
    
    # 1. 加载配置
    with open('Transformer_attention_pooling.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📁 配置文件: Transformer_attention_pooling.yaml")
    print(f"🤖 模型类型: {config['model']}")
    
    # 2. 获取模型参数
    complexity = config.get('model_complexity', 'low')
    if complexity in config['model_params']:
        mp = config['model_params'][complexity].copy()
    else:
        mp = config.get('model_params', {}).copy()
    
    mp['use_forecast'] = config.get('use_forecast', False)
    mp['past_hours'] = config['past_hours']
    mp['future_hours'] = config['future_hours']
    
    print(f"🔍 模型参数: {mp}")
    
    # 3. 创建模型
    print("\n🤖 创建模型...")
    model = ImprovedTransformer(
        hist_dim=1,  # 历史特征维度
        fcst_dim=11,  # 预测特征维度
        config=mp
    )
    print(f"✅ 模型创建成功")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 4. 创建测试数据
    print("\n📊 创建测试数据...")
    batch_size = 4
    past_hours = 24
    future_hours = 24
    
    # 创建随机测试数据
    hist_data = torch.randn(batch_size, past_hours, 1)
    fcst_data = torch.randn(batch_size, future_hours, 11)
    
    print(f"   历史数据形状: {hist_data.shape}")
    print(f"   预测数据形状: {fcst_data.shape}")
    
    # 5. 测试前向传播
    print("\n🔄 测试前向传播...")
    try:
        with torch.no_grad():
            output = model(hist_data, fcst_data)
            print(f"✅ 前向传播成功")
            print(f"   输出形状: {output.shape}")
            print(f"   输出范围: {output.min().item():.3f} - {output.max().item():.3f}")
            print(f"   输出均值: {output.mean().item():.3f}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 测试训练模式
    print("\n🎯 测试训练模式...")
    try:
        model.train()
        output = model(hist_data, fcst_data)
        print(f"✅ 训练模式成功")
        print(f"   输出形状: {output.shape}")
        
        # 计算损失
        target = torch.randn(batch_size, future_hours)
        loss = torch.nn.functional.mse_loss(output, target)
        print(f"   损失值: {loss.item():.3f}")
        
    except Exception as e:
        print(f"❌ 训练模式失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 简单测试完成！")

if __name__ == "__main__":
    main()
