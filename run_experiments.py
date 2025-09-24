#!/usr/bin/env python3
"""
实验运行脚本 - 用于对比不同Transformer架构的性能
"""

import os
import subprocess
import time
from datetime import datetime

def run_experiment(config_file, experiment_name):
    """运行单个实验"""
    print(f"\n🚀 开始实验: {experiment_name}")
    print(f"📁 配置文件: {config_file}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # 运行实验
        cmd = [
            "python", "main.py",
            "--config", config_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 实验 {experiment_name} 成功完成")
            print(f"⏱️  耗时: {duration:.2f} 秒")
            print("📊 输出:")
            print(result.stdout[-500:])  # 显示最后500个字符
        else:
            print(f"❌ 实验 {experiment_name} 失败")
            print(f"⏱️  耗时: {duration:.2f} 秒")
            print("❌ 错误信息:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 实验 {experiment_name} 超时 (1小时)")
    except Exception as e:
        print(f"💥 实验 {experiment_name} 出现异常: {str(e)}")

def main():
    """主函数 - 运行所有实验"""
    print("🔬 Transformer架构对比实验")
    print("=" * 50)
    
    # 实验配置列表
    experiments = [
        {
            "config": "Transformer_low_PV_plus_NWP_24h_noTE.yaml",
            "name": "原始Transformer (Last Timestep)"
        },
        {
            "config": "Transformer_attention_pooling.yaml", 
            "name": "改进Transformer (注意力池化)"
        },
        {
            "config": "Transformer_improved_PV_plus_NWP_24h.yaml",
            "name": "改进Transformer (综合改进)"
        },
        {
            "config": "Transformer_hybrid.yaml",
            "name": "混合Transformer (Encoder-Decoder)"
        }
    ]
    
    # 检查配置文件是否存在
    missing_configs = []
    for exp in experiments:
        if not os.path.exists(exp["config"]):
            missing_configs.append(exp["config"])
    
    if missing_configs:
        print("❌ 以下配置文件不存在:")
        for config in missing_configs:
            print(f"   - {config}")
        return
    
    # 运行所有实验
    total_start = time.time()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"实验 {i}/{len(experiments)}: {exp['name']}")
        print(f"{'='*60}")
        
        run_experiment(exp["config"], exp["name"])
        
        # 实验间暂停
        if i < len(experiments):
            print("\n⏸️  等待5秒后开始下一个实验...")
            time.sleep(5)
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    print(f"\n🎉 所有实验完成!")
    print(f"⏱️  总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"📊 结果保存在各自的results目录中")

def run_single_experiment(config_file):
    """运行单个实验的便捷函数"""
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    experiment_name = os.path.splitext(config_file)[0]
    run_experiment(config_file, experiment_name)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 运行指定的单个实验
        config_file = sys.argv[1]
        run_single_experiment(config_file)
    else:
        # 运行所有实验
        main()
