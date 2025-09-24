# Transformer-Enhanced 太阳能发电预测系统

## 项目概述

这是一个基于Transformer的太阳能发电预测系统，支持多种改进的架构和训练策略。项目实现了原始Transformer和改进版本的对比实验，用于评估不同架构在太阳能预测任务上的性能。

## 主要特性

### 🏗️ 模型架构
- **原始Transformer**: 使用最后时间步池化的基准模型
- **改进Transformer**: 支持多种池化策略的增强版本
- **混合Transformer**: 结合Encoder-Decoder和Encoder-Only优势的混合架构

### 🎯 池化策略
- **Last Timestep**: 原始方法，使用最后时间步
- **Mean Pooling**: 平均池化
- **Max Pooling**: 最大池化
- **Attention Pooling**: 注意力池化
- **Learned Attention**: 学习到的注意力权重

### 🚀 训练优化
- **AdamW优化器**: 更好的泛化能力
- **学习率调度**: ReduceLROnPlateau自适应调整
- **早停机制**: 防止过拟合
- **梯度裁剪**: 稳定训练过程
- **动态位置编码**: 支持更长序列

## 快速开始

### 环境要求
```bash
pip install torch numpy pandas scikit-learn pyyaml
```

### 运行测试
```bash
python test_improvements.py
```

### 运行单个实验
```bash
# 原始Transformer
python main.py --config Transformer_low_PV_plus_NWP_24h_noTE.yaml

# 改进Transformer (注意力池化)
python main.py --config Transformer_attention_pooling.yaml

# 改进Transformer (综合改进)
python main.py --config Transformer_improved_PV_plus_NWP_24h.yaml

# 混合Transformer
python main.py --config Transformer_hybrid.yaml
```

### 运行所有对比实验
```bash
python run_experiments.py
```

## 配置文件说明

### 1. Transformer_low_PV_plus_NWP_24h_noTE.yaml
- **用途**: 原始Transformer基准模型
- **特点**: 使用最后时间步池化
- **数据**: PV + NWP预测数据

### 2. Transformer_attention_pooling.yaml
- **用途**: 注意力池化改进
- **特点**: 使用注意力机制聚合序列信息
- **优势**: 更好地利用全局信息

### 3. Transformer_improved_PV_plus_NWP_24h.yaml
- **用途**: 综合改进版本
- **特点**: 包含所有改进策略
- **优势**: 最佳性能预期

### 4. Transformer_hybrid.yaml
- **用途**: 混合架构
- **特点**: 结合Encoder-Decoder优势
- **优势**: 更复杂的建模能力

## 实验对比

### 性能指标
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **训练时间**: 收敛速度
- **推理时间**: 预测效率

### 预期改进
- **预测精度**: RMSE/MAE降低5-15%
- **训练稳定性**: 减少周期性波动
- **收敛速度**: 提升20-30%
- **泛化能力**: 更好的验证集性能

## 项目结构

```
Transformer-enhanced/
├── main.py                          # 主程序入口
├── transformer.py                   # 原始Transformer实现
├── transformer_improved.py          # 改进的Transformer实现
├── train_dl.py                      # 深度学习训练代码
├── data_utils.py                    # 数据处理工具
├── eval_utils.py                    # 评估工具
├── train_utils.py                   # 训练工具
├── gpu_utils.py                     # GPU监控工具
├── run_experiments.py               # 实验运行脚本
├── test_improvements.py             # 测试脚本
├── *.yaml                           # 配置文件
├── IMPROVEMENTS.md                  # 改进说明
└── README.md                        # 项目说明
```

## 技术细节

### 改进的Transformer架构
```python
class ImprovedTransformer(nn.Module):
    def __init__(self, hist_dim, fcst_dim, config):
        # 输入投影层
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.fcst_proj = nn.Linear(fcst_dim, d_model)
        
        # 位置编码
        self.pos_enc = LocalPositionalEncoding(d_model)
        
        # Transformer编码器
        self.encoder = nn.TransformerEncoder(...)
        
        # 注意力池化
        self.attention_pool = AttentionPooling(d_model)
        
        # 改进的输出头
        self.output_head = nn.Sequential(...)
```

### 注意力池化机制
```python
class AttentionPooling(nn.Module):
    def forward(self, x):
        # 自注意力池化
        attended, _ = self.attention(query, x, x)
        return attended.squeeze(1)
```

## 结果分析

### 实验输出
每个实验的结果保存在独立的目录中：
- `results/ablation/`: 原始模型结果
- `results/attention_pooling/`: 注意力池化结果
- `results/improved_ablation/`: 综合改进结果
- `results/hybrid_transformer/`: 混合架构结果

### 结果文件
- `predictions.csv`: 预测结果
- `training_log.csv`: 训练日志
- `results.xlsx`: Excel格式结果汇总

## 自定义实验

### 创建新配置
1. 复制现有配置文件
2. 修改模型参数
3. 调整训练设置
4. 运行实验

### 添加新池化策略
1. 在`ImprovedTransformer`中添加新方法
2. 更新`pooling_type`选项
3. 测试新策略

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请创建Issue或联系项目维护者。

---

**注意**: 这是一个研究项目，用于对比不同Transformer架构在太阳能预测任务上的性能。所有改进都经过测试验证，可以安全使用。
