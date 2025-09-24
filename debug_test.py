#!/usr/bin/env python3
"""
调试测试脚本 - 逐步测试各个组件
"""

import yaml
import torch
import numpy as np
import pandas as pd

def test_config_loading():
    """测试配置文件加载"""
    print("🧪 测试配置文件加载...")
    
    try:
        with open('Transformer_low_PV_plus_NWP_24h_noTE.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功")
        print(f"   模型: {config['model']}")
        print(f"   复杂度: {config['model_complexity']}")
        print(f"   epoch_params: {config['epoch_params']}")
        print(f"   model_params: {config['model_params'].keys()}")
        
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def test_data_loading():
    """测试数据加载"""
    print("\n🧪 测试数据加载...")
    
    try:
        from data_utils import load_raw_data
        df = load_raw_data('Project1140.csv')
        print(f"✅ 数据加载成功: {len(df)} 行")
        
        # 测试数据预处理
        from data_utils import preprocess_features
        df_proj = df[df['ProjectID'] == 1140.0]
        print(f"   项目1140数据: {len(df_proj)} 行")
        
        if len(df_proj) > 0:
            config = {'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 
                     'weather_category': 'ablation_11_features', 'use_time_encoding': False}
            
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
                preprocess_features(df_proj, config)
            
            print(f"✅ 数据预处理成功")
            print(f"   历史特征: {len(hist_feats)} 个")
            print(f"   预测特征: {len(fcst_feats)} 个")
            print(f"   清理后数据: {len(df_clean)} 行")
            
            return df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target
        else:
            print("❌ 项目1140没有数据")
            return None
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def test_model_creation(config):
    """测试模型创建"""
    print("\n🧪 测试模型创建...")
    
    try:
        from transformer import Transformer
        
        # 模拟参数
        hist_dim = 10
        fcst_dim = 5
        mp = config['model_params']['low']
        mp['use_forecast'] = config.get('use_forecast', False)
        mp['past_hours'] = config['past_hours']
        mp['future_hours'] = config['future_hours']
        
        model = Transformer(hist_dim, fcst_dim, mp)
        print(f"✅ 模型创建成功")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试前向传播
        batch_size = 2
        hist_data = torch.randn(batch_size, config['past_hours'], hist_dim)
        fcst_data = torch.randn(batch_size, config['future_hours'], fcst_dim)
        
        with torch.no_grad():
            output = model(hist_data, fcst_data)
            print(f"✅ 前向传播成功: {output.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

def test_training_setup(config, model):
    """测试训练设置"""
    print("\n🧪 测试训练设置...")
    
    try:
        from train_utils import get_optimizer, get_scheduler
        
        train_params = config['train_params']
        opt = get_optimizer(model, lr=float(train_params['learning_rate']))
        sched = get_scheduler(opt, train_params)
        
        print(f"✅ 优化器和调度器创建成功")
        
        # 测试epoch参数获取
        complexity = config.get('model_complexity', 'low')
        epoch_params = config.get('epoch_params', {})
        
        if complexity in epoch_params and 'epochs' in epoch_params[complexity]:
            epochs = epoch_params[complexity]['epochs']
        else:
            epochs = 50
            
        print(f"✅ Epoch参数获取成功: {epochs}")
        
        return opt, sched, epochs
        
    except Exception as e:
        print(f"❌ 训练设置失败: {e}")
        return None, None, None

def main():
    """主测试函数"""
    print("🔬 逐步调试测试")
    print("=" * 50)
    
    # 1. 测试配置加载
    config = test_config_loading()
    if not config:
        return
    
    # 2. 测试数据加载
    data_result = test_data_loading()
    if not data_result:
        return
    
    # 3. 测试模型创建
    model = test_model_creation(config)
    if not model:
        return
    
    # 4. 测试训练设置
    opt, sched, epochs = test_training_setup(config, model)
    if not opt:
        return
    
    print("\n🎉 所有组件测试通过！")
    print("🚀 可以尝试运行完整实验")

if __name__ == "__main__":
    main()
