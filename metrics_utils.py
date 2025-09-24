"""
评估指标计算工具
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_mse(y_true, y_pred):
    """计算均方误差"""
    return mean_squared_error(y_true, y_pred)

def calculate_metrics(y_true, y_pred):
    """计算所有评估指标"""
    # 展平数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 基本指标
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # 归一化指标
    y_mean = np.mean(y_true_flat)
    nrmse = (rmse / (y_mean + 1e-8)) * 100 if y_mean > 0 else 0
    
    # SMAPE (对称平均绝对百分比误差)
    smape = np.mean(2 * np.abs(y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + np.abs(y_pred_flat) + 1e-8)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'r_square': r2,  # 别名
        'nrmse': nrmse,
        'smape': smape
    }
