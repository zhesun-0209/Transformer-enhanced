# Transformer架构改进说明

## 概述

本项目对原始Transformer架构进行了多项改进，以提高太阳能发电预测的性能。主要改进包括模型架构优化、训练策略增强和数据处理改进。

## 主要改进

### 1. 模型架构改进

#### 1.1 注意力池化 (Attention Pooling)
- **原始方法**: 使用最后时间步的表示
- **改进方法**: 使用注意力机制聚合整个序列的信息
- **优势**: 更好地利用历史序列的全局信息

```python
# 原始方法
last_timestep = combined[:, -1, :]  # 只使用最后时间步

# 改进方法
global_repr = self.attention_pool(combined)  # 注意力池化
```

#### 1.2 多种池化策略
- **Last Timestep**: 原始方法
- **Mean Pooling**: 平均池化
- **Max Pooling**: 最大池化  
- **Attention Pooling**: 注意力池化
- **Learned Attention**: 学习到的注意力权重

#### 1.3 混合架构 (HybridTransformer)
- 结合Encoder-Decoder和Encoder-Only的优势
- 使用解码器处理预测数据
- 保持一次性预测的效率

### 2. 训练策略改进

#### 2.1 优化器改进
- **原始**: Adam
- **改进**: AdamW (权重衰减=0.01)
- **优势**: 更好的泛化能力

#### 2.2 学习率调度
- **原始**: 固定学习率
- **改进**: ReduceLROnPlateau
- **参数**: factor=0.5, patience=8, min_lr=1e-6

#### 2.3 早停机制
- **patience**: 15 epochs
- **监控**: 验证损失
- **优势**: 防止过拟合，节省计算资源

#### 2.4 梯度裁剪
- **max_norm**: 1.0
- **优势**: 稳定训练，防止梯度爆炸

#### 2.5 批次大小优化
- **原始**: 32
- **改进**: 64 (至少)
- **优势**: 更稳定的梯度估计

### 3. 数据处理改进

#### 3.1 动态位置编码
- **原始**: 固定长度位置编码
- **改进**: 动态扩展位置编码
- **优势**: 支持更长序列 (168小时)

#### 3.2 特征工程
- 时间编码特征 (月份、小时的正弦/余弦)
- 天气特征分类 (辐射、温度、湿度等)
- 预测数据支持 (NWP)

### 4. 输出层改进

#### 4.1 激活函数
- **原始**: GELU
- **改进**: ReLU + Sigmoid
- **优势**: 更稳定的训练，输出范围[0,1]

#### 4.2 网络结构
- **原始**: 单层MLP
- **改进**: 多层MLP with Dropout
- **优势**: 更强的表达能力

## 实验配置

### 配置文件说明

1. **Transformer_low_PV_plus_NWP_24h_noTE.yaml**
   - 原始Transformer架构
   - 使用最后时间步池化
   - 基准对比

2. **Transformer_attention_pooling.yaml**
   - 改进Transformer + 注意力池化
   - 测试注意力池化的效果

3. **Transformer_improved_PV_plus_NWP_24h.yaml**
   - 综合改进版本
   - 包含所有改进策略

4. **Transformer_hybrid.yaml**
   - 混合架构
   - 测试Encoder-Decoder优势

### 运行实验

#### 运行所有实验
```bash
python run_experiments.py
```

#### 运行单个实验
```bash
python run_experiments.py Transformer_attention_pooling.yaml
```

#### 手动运行
```bash
python main.py --config Transformer_improved_PV_plus_NWP_24h.yaml
```

## 预期改进效果

### 性能指标
- **RMSE**: 预期降低5-15%
- **MAE**: 预期降低5-15%
- **R²**: 预期提升5-10%
- **训练稳定性**: 显著提升

### 训练效率
- **收敛速度**: 预期提升20-30%
- **训练稳定性**: 减少周期性波动
- **泛化能力**: 更好的验证集性能

## 结果分析

### 对比维度
1. **预测精度**: RMSE, MAE, R²
2. **训练效率**: 收敛时间, 最佳epoch
3. **模型复杂度**: 参数数量, 推理时间
4. **稳定性**: 训练曲线平滑度

### 结果保存
- 每个实验的结果保存在独立的目录中
- 包含预测结果、训练日志、性能指标
- 支持Excel格式导出，便于分析

## 技术细节

### 注意力池化实现
```python
class AttentionPooling(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attended, _ = self.attention(query, x, x)
        return attended.squeeze(1)
```

### 混合架构实现
```python
class HybridTransformer(nn.Module):
    def __init__(self, hist_dim, fcst_dim, config):
        # 编码器处理历史数据
        self.encoder = nn.TransformerEncoder(...)
        # 解码器处理预测数据
        self.decoder = nn.TransformerDecoder(...)
        # 未来时间步嵌入
        self.future_embedding = nn.Parameter(...)
```

## 后续改进方向

1. **注意力机制优化**
   - 因果注意力
   - 局部注意力
   - 稀疏注意力

2. **模型压缩**
   - 知识蒸馏
   - 模型剪枝
   - 量化优化

3. **多任务学习**
   - 同时预测多个时间尺度
   - 不确定性估计
   - 异常检测

4. **在线学习**
   - 增量学习
   - 模型适应
   - 实时更新
