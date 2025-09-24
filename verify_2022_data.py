#!/usr/bin/env python3
"""
验证2022年数据使用情况
"""

import yaml
import pandas as pd
from data_utils import load_raw_data, preprocess_features, create_sliding_windows

def main():
    print("🔍 验证2022年数据使用情况")
    print("=" * 50)
    
    # 1. 检查配置文件
    with open('Transformer_low_PV_plus_NWP_24h_noTE.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📅 配置日期范围: {config['start_date']} 到 {config['end_date']}")
    
    # 2. 加载原始数据
    df = load_raw_data('Project1140.csv')
    df_proj = df[df['ProjectID'] == 1140.0]
    
    print(f"📊 项目1140总数据: {len(df_proj)} 行")
    print(f"📅 原始数据日期范围: {df_proj['Datetime'].min()} 到 {df_proj['Datetime'].max()}")
    
    # 3. 数据预处理
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = \
        preprocess_features(df_proj, config)
    
    print(f"📊 预处理后数据: {len(df_clean)} 行")
    print(f"📅 预处理后日期范围: {df_clean['Datetime'].min()} 到 {df_clean['Datetime'].max()}")
    
    # 4. 验证日期过滤
    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    
    if len(df_clean) > 0:
        actual_start = df_clean['Datetime'].min()
        actual_end = df_clean['Datetime'].max()
        
        print(f"\n✅ 日期过滤验证:")
        print(f"   配置开始日期: {start_date}")
        print(f"   实际开始日期: {actual_start}")
        print(f"   配置结束日期: {end_date}")
        print(f"   实际结束日期: {actual_end}")
        
        if actual_start >= start_date and actual_end <= end_date:
            print("✅ 日期过滤正确！")
        else:
            print("❌ 日期过滤有问题！")
    
    # 5. 创建滑动窗口测试
    try:
        X_hist, X_fcst, y, hours, dates = create_sliding_windows(
            df_clean,
            past_hours=config['past_hours'],
            future_hours=config['future_hours'],
            hist_feats=hist_feats,
            fcst_feats=fcst_feats,
            no_hist_power=not config.get('use_pv', True)
        )
        
        print(f"\n✅ 滑动窗口创建成功:")
        print(f"   历史数据形状: {X_hist.shape}")
        print(f"   预测数据形状: {X_fcst.shape}")
        print(f"   目标数据形状: {y.shape}")
        print(f"   样本数量: {len(dates)}")
        
        if len(dates) > 0:
            print(f"   第一个样本日期: {dates[0]}")
            print(f"   最后一个样本日期: {dates[-1]}")
        
    except Exception as e:
        print(f"❌ 滑动窗口创建失败: {e}")
    
    # 6. 数据统计
    if len(df_clean) > 0:
        print(f"\n📈 数据统计:")
        print(f"   总天数: {len(df_clean['Datetime'].dt.date.unique())}")
        print(f"   平均每天样本数: {len(df_clean) / len(df_clean['Datetime'].dt.date.unique()):.1f}")
        print(f"   Capacity Factor范围: {df_clean['Capacity Factor'].min():.2f} - {df_clean['Capacity Factor'].max():.2f}")
        print(f"   Capacity Factor均值: {df_clean['Capacity Factor'].mean():.2f}")
    
    print(f"\n🎉 2022年数据验证完成！")

if __name__ == "__main__":
    main()
