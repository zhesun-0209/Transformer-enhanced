#!/usr/bin/env python3
"""
直接测试is_dl逻辑
"""

import yaml

def test_is_dl_logic():
    print("🔍 直接测试is_dl逻辑")
    print("=" * 50)
    
    # 加载配置
    with open('Transformer_attention_pooling.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📁 配置文件: Transformer_attention_pooling.yaml")
    print(f"🤖 模型名称: {repr(config['model'])}")
    
    # 测试is_dl逻辑 - 完全复制main.py中的逻辑
    is_dl = config["model"] in ["Transformer", "ImprovedTransformer", "HybridTransformer", "LSTM", "GRU", "TCN"]
    
    print(f"🔍 is_dl逻辑测试:")
    print(f"   模型名称: {config['model']}")
    print(f"   是否在DL列表中: {config['model'] in ['Transformer', 'ImprovedTransformer', 'HybridTransformer', 'LSTM', 'GRU', 'TCN']}")
    print(f"   is_dl结果: {is_dl}")
    print(f"   模型类型: {'DL' if is_dl else 'ML'}")
    
    # 测试字符串比较
    print(f"\n🔍 字符串比较测试:")
    model_name = config['model']
    print(f"   模型名称: {repr(model_name)}")
    print(f"   是否等于'ImprovedTransformer': {model_name == 'ImprovedTransformer'}")
    print(f"   是否等于'ImprovedTransformer' (repr): {repr(model_name) == repr('ImprovedTransformer')}")
    
    # 测试列表包含
    print(f"\n🔍 列表包含测试:")
    dl_models = ["Transformer", "ImprovedTransformer", "HybridTransformer", "LSTM", "GRU", "TCN"]
    print(f"   DL模型列表: {dl_models}")
    print(f"   模型是否在列表中: {model_name in dl_models}")
    
    # 测试每个模型
    print(f"\n🔍 逐个测试:")
    for model in dl_models:
        is_match = model_name == model
        print(f"   {model}: {is_match}")
    
    return is_dl

if __name__ == "__main__":
    result = test_is_dl_logic()
    print(f"\n🎉 最终结果: is_dl = {result}")
