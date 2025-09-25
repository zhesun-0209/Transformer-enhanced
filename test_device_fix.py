#!/usr/bin/env python3
"""
测试设备修复
"""

import yaml
import torch
from data_utils import load_raw_data, preprocess_features, create_sliding_windows
from transformer import Transformer

def main():
    print("🔍 测试设备修复")
    print("=" * 50)
    
    # 1. 加载配置
    with open('Transformer_low_PV_plus_NWP_24h_noTE.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📁 配置文件: Transformer_low_PV_plus_NWP_24h_noTE.yaml")
    print(f"🤖 模型类型: {config['model']}")
    
    # 2. 加载数据
    print("\n📊 加载数据...")
    df = load_raw_data('Project1140.csv')
    df_proj = df[df['ProjectID'] == 1140.0]
    
    # 3. 预处理
    print("🔧 预处理数据...")
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
        preprocess_features(df_proj, config)
    
    print(f"✅ 预处理完成: {len(df_clean)} 行")
    
    # 4. 创建滑动窗口
    print("\n🪟 创建滑动窗口...")
    X_hist, X_fcst, y, hours, dates = create_sliding_windows(
        df_clean,
        past_hours=config['past_hours'],
        future_hours=config['future_hours'],
        hist_feats=hist_feats,
        fcst_feats=fcst_feats,
        no_hist_power=not config.get('use_pv', True)
    )
    
    print(f"✅ 滑动窗口创建成功:")
    print(f"   历史数据形状: {X_hist.shape}")
    print(f"   预测数据形状: {X_fcst.shape}")
    print(f"   目标数据形状: {y.shape}")
    
    # 5. 测试模型创建和前向传播
    print("\n🤖 测试模型...")
    try:
        # 获取模型参数
        complexity = config.get('model_complexity', 'low')
        if complexity in config['model_params']:
            mp = config['model_params'][complexity].copy()
        else:
            mp = config.get('model_params', {}).copy()
        
        mp['use_forecast'] = config.get('use_forecast', False)
        mp['past_hours'] = config['past_hours']
        mp['future_hours'] = config['future_hours']
        
        # 创建模型
        model = Transformer(
            hist_dim=len(hist_feats),
            fcst_dim=len(fcst_feats),
            config=mp
        )
        
        print(f"✅ 模型创建成功")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试前向传播
        print("\n🔄 测试前向传播...")
        with torch.no_grad():
            # 使用小批量测试
            batch_size = min(4, X_hist.shape[0])
            hist_batch = torch.tensor(X_hist[:batch_size], dtype=torch.float32)
            fcst_batch = torch.tensor(X_fcst[:batch_size], dtype=torch.float32)
            
            output = model(hist_batch, fcst_batch)
            print(f"✅ 前向传播成功")
            print(f"   输入形状: hist={hist_batch.shape}, fcst={fcst_batch.shape}")
            print(f"   输出形状: {output.shape}")
            print(f"   输出设备: {output.device}")
            print(f"   输出范围: {output.min().item():.3f} - {output.max().item():.3f}")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 设备修复测试完成！")

if __name__ == "__main__":
    main()
