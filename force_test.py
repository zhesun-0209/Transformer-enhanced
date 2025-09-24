#!/usr/bin/env python3
"""
强制测试 - 直接复制main.py的逻辑
"""

import os
import yaml
import pandas as pd
import torch
from copy import deepcopy
from data_utils import load_raw_data, preprocess_features, create_sliding_windows

def main():
    print("🔍 强制测试 - 复制main.py逻辑")
    print("=" * 50)
    
    # === 步骤1：加载配置 ===
    with open('Transformer_attention_pooling.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📁 配置文件: Transformer_attention_pooling.yaml")
    print(f"🤖 原始模型名称: {repr(config['model'])}")
    
    # === 步骤2：完全复制main.py中的is_dl逻辑 ===
    is_dl = config["model"] in ["Transformer", "ImprovedTransformer", "HybridTransformer", "LSTM", "GRU", "TCN"]
    alg_type = "dl" if is_dl else "ml"
    
    print(f"🔍 is_dl计算:")
    print(f"   is_dl = {is_dl}")
    print(f"   alg_type = {alg_type}")
    print(f"   模型类型: {'DL' if is_dl else 'ML'}")
    
    # === 步骤3：模拟项目循环 ===
    print(f"\n📊 加载数据...")
    df = load_raw_data('Project1140.csv')
    
    # === 模拟项目循环内部逻辑 ===
    for pid in [1140.0]:  # 只测试一个项目
        print(f"\n🏭 处理项目 {pid}")
        df_proj = df[df["ProjectID"] == pid]
        
        # === 步骤4：预处理数据 ===
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
            preprocess_features(df_proj, config)
        
        print(f"   ✅ 预处理完成: {len(df_clean)} 行")
        
        # === 步骤5：创建滑动窗口 ===
        X_hist, X_fcst, y, hours, dates = create_sliding_windows(
            df_clean,
            past_hours=config['past_hours'],
            future_hours=config['future_hours'],
            hist_feats=hist_feats,
            fcst_feats=fcst_feats,
            no_hist_power=not config.get('use_pv', True)
        )
        
        print(f"   ✅ 滑动窗口创建成功: {X_hist.shape}")
        
        # === 步骤6：模拟main.py中的配置复制和检查 ===
        plant_save_dir = config["save_dir"]
        os.makedirs(plant_save_dir, exist_ok=True)
        cfg = deepcopy(config)
        cfg["save_dir"] = plant_save_dir
        
        print(f"   📋 配置复制后:")
        print(f"      cfg['model'] = {repr(cfg.get('model', 'NOT_FOUND'))}")
        print(f"      is_dl = {is_dl}")
        print(f"      模型类型: {'DL' if is_dl else 'ML'}")
        
        # === 步骤7：检查条件判断 ===
        print(f"   🔍 条件判断:")
        print(f"      if is_dl: {is_dl}")
        if is_dl:
            print(f"      ✅ 会调用train_dl_model")
        else:
            print(f"      ❌ 会调用train_ml_model (错误)")
        
        break  # 只测试一个项目
    
    print(f"\n🎉 强制测试完成！")

if __name__ == "__main__":
    main()
