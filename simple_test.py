#!/usr/bin/env python3
"""
简单测试脚本 - 最小化运行
"""

import yaml
import torch
import numpy as np
from data_utils import load_raw_data, preprocess_features, create_sliding_windows, split_data
from transformer import Transformer
from train_utils import get_optimizer, get_scheduler

def main():
    print("🧪 简单测试开始...")
    
    try:
        # 1. 加载配置
        with open('Transformer_low_PV_plus_NWP_24h_noTE.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ 配置加载成功")
        
        # 2. 加载数据
        df = load_raw_data('Project1140.csv')
        df_proj = df[df['ProjectID'] == 1140.0]
        print(f"✅ 数据加载成功: {len(df_proj)} 行")
        
        # 3. 预处理
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
            preprocess_features(df_proj, config)
        print(f"✅ 预处理成功: {len(df_clean)} 行")
        
        # 4. 创建滑动窗口
        X_hist, X_fcst, y, hours, dates = create_sliding_windows(
            df_clean,
            past_hours=config['past_hours'],
            future_hours=config['future_hours'],
            hist_feats=hist_feats,
            fcst_feats=fcst_feats,
            no_hist_power=not config.get('use_pv', True)
        )
        print(f"✅ 滑动窗口创建成功: {X_hist.shape}")
        
        # 5. 数据分割
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
        Xh_va, Xf_va, y_va, hrs_va, dates_va, \
        Xh_te, Xf_te, y_te, hrs_te, dates_te = split_data(
            X_hist, X_fcst, y, hours, dates,
            train_ratio=0.8,
            val_ratio=0.1
        )
        print(f"✅ 数据分割成功: 训练集 {Xh_tr.shape[0]} 样本")
        
        # 6. 创建模型
        mp = config['model_params']['low']
        mp['use_forecast'] = config.get('use_forecast', False)
        mp['past_hours'] = config['past_hours']
        mp['future_hours'] = config['future_hours']
        
        model = Transformer(Xh_tr.shape[2], Xf_tr.shape[2], mp)
        print(f"✅ 模型创建成功: {sum(p.numel() for p in model.parameters())} 参数")
        
        # 7. 测试前向传播
        with torch.no_grad():
            test_input_hist = torch.randn(2, Xh_tr.shape[1], Xh_tr.shape[2])
            test_input_fcst = torch.randn(2, Xf_tr.shape[1], Xf_tr.shape[2])
            output = model(test_input_hist, test_input_fcst)
            print(f"✅ 前向传播成功: {output.shape}")
        
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
