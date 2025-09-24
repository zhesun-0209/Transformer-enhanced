"""
Excel结果保存工具
"""
import os
import pandas as pd
from datetime import datetime

def save_plant_excel_results(plant_id, result_data, save_dir):
    """保存厂级Excel结果"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建结果数据
    config = result_data['config']
    metrics = result_data['metrics']
    
    # 创建结果DataFrame
    result_df = pd.DataFrame([{
        'plant_id': plant_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': config['model'],
        'use_pv': config['use_pv'],
        'use_hist_weather': config['use_hist_weather'],
        'use_forecast': config['use_forecast'],
        'weather_category': config['weather_category'],
        'use_time_encoding': config['use_time_encoding'],
        'past_days': config['past_days'],
        'model_complexity': config['model_complexity'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'train_time_sec': metrics['train_time_sec'],
        'inference_time_sec': metrics['inference_time_sec'],
        'param_count': metrics['param_count'],
        'samples_count': metrics['samples_count'],
        'mse': metrics['mse'],
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'nrmse': metrics['nrmse'],
        'r_square': metrics['r_square'],
        'smape': metrics['smape'],
        'best_epoch': metrics['best_epoch'],
        'final_lr': metrics['final_lr'],
        'gpu_memory_used': metrics['gpu_memory_used']
    }])
    
    # 保存到Excel文件
    excel_file = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    result_df.to_excel(excel_file, index=False)
    
    return excel_file

def append_plant_excel_results(plant_id, result, save_dir):
    """追加厂级Excel结果"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建结果数据
    config = result['config']
    metrics = result['metrics']
    
    # 创建结果DataFrame
    result_df = pd.DataFrame([{
        'plant_id': plant_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': config['model'],
        'use_pv': config['use_pv'],
        'use_hist_weather': config['use_hist_weather'],
        'use_forecast': config['use_forecast'],
        'weather_category': config['weather_category'],
        'use_time_encoding': config['use_time_encoding'],
        'past_days': config['past_days'],
        'model_complexity': config['model_complexity'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'train_time_sec': metrics['train_time_sec'],
        'inference_time_sec': metrics['inference_time_sec'],
        'param_count': metrics['param_count'],
        'samples_count': metrics['samples_count'],
        'mse': metrics['mse'],
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'nrmse': metrics['nrmse'],
        'r_square': metrics['r_square'],
        'smape': metrics['smape'],
        'best_epoch': metrics['best_epoch'],
        'final_lr': metrics['final_lr'],
        'gpu_memory_used': metrics['gpu_memory_used']
    }])
    
    # 保存到CSV文件（追加模式）
    csv_file = os.path.join(save_dir, "results.csv")
    
    if os.path.exists(csv_file):
        # 如果文件存在，追加数据
        existing_df = pd.read_csv(csv_file)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    
    result_df.to_csv(csv_file, index=False)
    
    return csv_file
