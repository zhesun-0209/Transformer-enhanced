#!/usr/bin/env python3
"""
测试所有模块导入是否正常
"""

def test_imports():
    """测试所有模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 测试基础模块
        import torch
        import numpy as np
        import pandas as pd
        import yaml
        print("✅ 基础依赖包导入成功")
        
        # 测试项目模块
        from data_utils import load_raw_data, preprocess_features
        print("✅ data_utils导入成功")
        
        from train_utils import get_optimizer, get_scheduler
        print("✅ train_utils导入成功")
        
        from metrics_utils import calculate_metrics, calculate_mse
        print("✅ metrics_utils导入成功")
        
        from excel_utils import save_plant_excel_results
        print("✅ excel_utils导入成功")
        
        from gpu_utils import get_gpu_memory_used
        print("✅ gpu_utils导入成功")
        
        from transformer import Transformer
        print("✅ transformer导入成功")
        
        from transformer_improved import ImprovedTransformer, HybridTransformer
        print("✅ transformer_improved导入成功")
        
        from train_dl import train_dl_model
        print("✅ train_dl导入成功")
        
        from eval_utils import save_results
        print("✅ eval_utils导入成功")
        
        from main import main
        print("✅ main导入成功")
        
        print("\n🎉 所有模块导入成功！")
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

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
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ {config_file} 加载成功")
        except Exception as e:
            print(f"❌ {config_file} 加载失败: {e}")

def test_data_loading():
    """测试数据加载"""
    print("\n🧪 测试数据加载...")
    
    try:
        from data_utils import load_raw_data
        df = load_raw_data('Project1140.csv')
        print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")
        print(f"   列名: {list(df.columns)[:5]}...")
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔬 模块导入和数据加载测试")
    print("=" * 50)
    
    # 测试导入
    import_success = test_imports()
    
    # 测试配置
    test_config_loading()
    
    # 测试数据
    data_success = test_data_loading()
    
    if import_success and data_success:
        print("\n🎉 所有测试通过！可以开始实验了。")
        print("🚀 运行命令: python run_experiments.py")
    else:
        print("\n❌ 测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
