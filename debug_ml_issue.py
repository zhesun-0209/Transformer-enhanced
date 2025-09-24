#!/usr/bin/env python3
"""
调试ML问题
"""

import yaml
import sys
import os

def main():
    print("🔍 调试ML问题")
    print("=" * 50)
    
    # 1. 检查配置文件
    with open('Transformer_attention_pooling.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📁 配置文件: Transformer_attention_pooling.yaml")
    print(f"🤖 模型名称: {repr(config['model'])}")
    print(f"📝 模型类型: {type(config['model'])}")
    
    # 2. 测试is_dl逻辑
    dl_models = ["Transformer", "ImprovedTransformer", "HybridTransformer", "LSTM", "GRU", "TCN"]
    is_dl = config["model"] in dl_models
    
    print(f"\n🔍 is_dl逻辑测试:")
    print(f"   DL模型列表: {dl_models}")
    print(f"   当前模型: {config['model']}")
    print(f"   是否在列表中: {config['model'] in dl_models}")
    print(f"   is_dl结果: {is_dl}")
    print(f"   模型类型: {'DL' if is_dl else 'ML'}")
    
    # 3. 检查main.py中的逻辑
    print(f"\n📄 检查main.py中的逻辑:")
    
    # 读取main.py文件
    with open('main.py', 'r') as f:
        main_content = f.read()
    
    # 查找is_dl定义
    lines = main_content.split('\n')
    for i, line in enumerate(lines):
        if 'is_dl = config["model"] in' in line:
            print(f"   第{i+1}行: {line.strip()}")
            # 检查下一行
            if i+1 < len(lines):
                print(f"   第{i+2}行: {lines[i+1].strip()}")
    
    # 4. 测试导入
    print(f"\n📦 测试导入:")
    try:
        # 重新导入main模块
        if 'main' in sys.modules:
            del sys.modules['main']
        import main
        print(f"   ✅ main模块导入成功")
    except Exception as e:
        print(f"   ❌ main模块导入失败: {e}")
    
    # 5. 检查是否有其他问题
    print(f"\n🔍 其他检查:")
    
    # 检查是否有字符串比较问题
    model_name = config['model']
    print(f"   模型名称长度: {len(model_name)}")
    print(f"   模型名称字节: {model_name.encode('utf-8')}")
    print(f"   是否等于'ImprovedTransformer': {model_name == 'ImprovedTransformer'}")
    print(f"   是否等于'ImprovedTransformer' (repr): {repr(model_name) == repr('ImprovedTransformer')}")
    
    # 6. 测试完整的判断逻辑
    print(f"\n🧪 完整判断逻辑测试:")
    
    # 模拟main.py中的逻辑
    complexity = config.get("model_complexity", "low")
    is_dl = config["model"] in ["Transformer", "ImprovedTransformer", "HybridTransformer", "LSTM", "GRU", "TCN"]
    
    print(f"   complexity: {complexity}")
    print(f"   is_dl: {is_dl}")
    print(f"   模型类型: {'DL' if is_dl else 'ML'}")
    
    if is_dl:
        print(f"   ✅ 应该使用train_dl_model")
    else:
        print(f"   ❌ 会使用train_ml_model (这会导致错误)")
    
    print(f"\n🎉 调试完成！")

if __name__ == "__main__":
    main()
