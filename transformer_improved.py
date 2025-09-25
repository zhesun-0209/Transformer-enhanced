"""
Improved Transformer forecasting model with enhanced architecture.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class LocalPositionalEncoding(nn.Module):
    """局部位置编码 - 支持更长的序列"""
    def __init__(self, d_model, max_len=200):  # 增加到200以支持168小时lookback
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 如果序列长度超过max_len，动态扩展位置编码
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # 动态创建更长的位置编码
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                               (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)
        else:
            return x + self.pe[:, :seq_len]

class AttentionPooling(nn.Module):
    """注意力池化层 - 改进的全局表示"""
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        # x: (B, seq_len, d_model)
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, d_model)
        
        # 自注意力池化
        attended, _ = self.attention(query, x, x)  # (B, 1, d_model)
        return attended.squeeze(1)  # (B, d_model)

class ImprovedTransformer(nn.Module):
    """改进的PV预测Transformer模型"""
    def __init__(
        self,
        hist_dim: int,
        fcst_dim: int,
        config: dict
    ):
        super().__init__()
        self.cfg = config
        d_model = config['d_model']

        # 输入投影层
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.fcst_proj = nn.Linear(fcst_dim, d_model) if fcst_dim > 0 else None

        # 位置编码
        self.pos_enc = LocalPositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['num_heads'],
            dim_feedforward=d_model * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # 改进的池化层 - 多种选择
        self.pooling_type = config.get('pooling_type', 'attention')  # 'last', 'mean', 'max', 'attention'
        
        if self.pooling_type == 'attention':
            self.attention_pool = AttentionPooling(d_model, num_heads=1)
        elif self.pooling_type == 'learned_attention':
            self.learned_attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )

        # 改进的输出头
        self.output_head = nn.Sequential(
            nn.Linear(d_model, config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, config['future_hours']),
            nn.Sigmoid()
        )

    def forward(
        self,
        hist: torch.Tensor,        # shape: (B, past_hours, hist_dim)
        fcst: torch.Tensor = None  # shape: (B, future_hours, fcst_dim), optional
    ) -> torch.Tensor:
        # 编码历史输入
        h = self.hist_proj(hist)              # (B, past_hours, d_model)
        h = self.pos_enc(h)
        h_enc = self.encoder(h)               # (B, past_hours, d_model)

        # 编码预测输入（如果适用）
        if self.cfg.get('use_forecast', False) and fcst is not None and self.fcst_proj is not None:
            f = self.fcst_proj(fcst)          # (B, future_hours, d_model)
            f = self.pos_enc(f)
            f_enc = self.encoder(f)           # (B, future_hours, d_model)
            
            # 改进：分别对历史和预测进行池化，然后融合
            # 这样既利用了历史信息，也利用了预测信息
            if self.pooling_type == 'last':
                hist_repr = h_enc[:, -1, :]       # (B, d_model) - 历史最后时间步
                fcst_repr = f_enc[:, -1, :]       # (B, d_model) - 预测最后时间步
            elif self.pooling_type == 'mean':
                hist_repr = torch.mean(h_enc, dim=1)  # (B, d_model) - 历史平均
                fcst_repr = torch.mean(f_enc, dim=1)  # (B, d_model) - 预测平均
            elif self.pooling_type == 'max':
                hist_repr = torch.max(h_enc, dim=1)[0]  # (B, d_model) - 历史最大
                fcst_repr = torch.max(f_enc, dim=1)[0]  # (B, d_model) - 预测最大
            elif self.pooling_type == 'attention':
                hist_repr = self.attention_pool(h_enc)  # (B, d_model) - 历史注意力池化
                fcst_repr = self.attention_pool(f_enc)  # (B, d_model) - 预测注意力池化
            elif self.pooling_type == 'learned_attention':
                hist_weights = self.learned_attention(h_enc)  # (B, past_hours, 1)
                hist_weights = F.softmax(hist_weights, dim=1)
                hist_repr = torch.sum(h_enc * hist_weights, dim=1)  # (B, d_model)
                
                fcst_weights = self.learned_attention(f_enc)  # (B, future_hours, 1)
                fcst_weights = F.softmax(fcst_weights, dim=1)
                fcst_repr = torch.sum(f_enc * fcst_weights, dim=1)  # (B, d_model)
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
            # 融合历史和预测表示
            global_repr = torch.cat([hist_repr, fcst_repr], dim=-1)  # (B, 2*d_model)
        else:
            # 只有历史信息时，使用原始池化方法
            if self.pooling_type == 'last':
                global_repr = h_enc[:, -1, :]  # (B, d_model)
            elif self.pooling_type == 'mean':
                global_repr = torch.mean(h_enc, dim=1)  # (B, d_model)
            elif self.pooling_type == 'max':
                global_repr = torch.max(h_enc, dim=1)[0]  # (B, d_model)
            elif self.pooling_type == 'attention':
                global_repr = self.attention_pool(h_enc)  # (B, d_model)
            elif self.pooling_type == 'learned_attention':
                attention_weights = self.learned_attention(h_enc)  # (B, seq_len, 1)
                attention_weights = F.softmax(attention_weights, dim=1)  # 归一化
                global_repr = torch.sum(h_enc * attention_weights, dim=1)  # (B, d_model)
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # 调整输出头以适应不同的输入维度
        if self.cfg.get('use_forecast', False) and fcst is not None and self.fcst_proj is not None:
            # 有预测信息时，输入维度是2*d_model
            if not hasattr(self, 'output_head_with_forecast'):
                self.output_head_with_forecast = nn.Sequential(
                    nn.Linear(2 * self.cfg['d_model'], self.cfg['hidden_dim']),
                    nn.ReLU(),
                    nn.Dropout(self.cfg['dropout']),
                    nn.Linear(self.cfg['hidden_dim'], self.cfg['future_hours']),
                    nn.Sigmoid()
                )
            result = self.output_head_with_forecast(global_repr)  # (B, future_hours)
        else:
            # 没有预测信息时，使用原始输出头
            result = self.output_head(global_repr)   # (B, future_hours)
        
        return result * 100  # 乘以100转换为百分比

class HybridTransformer(nn.Module):
    """混合架构Transformer - 结合Encoder-Decoder和Encoder-Only的优势"""
    def __init__(
        self,
        hist_dim: int,
        fcst_dim: int,
        config: dict
    ):
        super().__init__()
        self.cfg = config
        d_model = config['d_model']

        # 输入投影层
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.fcst_proj = nn.Linear(fcst_dim, d_model) if fcst_dim > 0 else None

        # 位置编码
        self.pos_enc = LocalPositionalEncoding(d_model)

        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['num_heads'],
            dim_feedforward=d_model * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # 解码器（用于处理预测数据）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config['num_heads'],
            dim_feedforward=d_model * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['num_layers'])

        # 未来时间步的嵌入
        self.future_embedding = nn.Parameter(torch.randn(config['future_hours'], d_model))

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(d_model, config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], 1),  # 每个时间步输出1个值
            nn.Sigmoid()
        )

    def forward(
        self,
        hist: torch.Tensor,        # shape: (B, past_hours, hist_dim)
        fcst: torch.Tensor = None  # shape: (B, future_hours, fcst_dim), optional
    ) -> torch.Tensor:
        # 编码历史输入
        h = self.hist_proj(hist)              # (B, past_hours, d_model)
        h = self.pos_enc(h)
        memory = self.encoder(h)              # (B, past_hours, d_model)

        # 创建未来时间步的查询
        batch_size = hist.size(0)
        future_queries = self.future_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # (B, future_hours, d_model)
        
        # 准备Key-Value信息
        if self.cfg.get('use_forecast', False) and fcst is not None and self.fcst_proj is not None:
            # 编码预测天气数据
            f = self.fcst_proj(fcst)          # (B, future_hours, d_model)
            f = self.pos_enc(f)
            f_enc = self.encoder(f)           # (B, future_hours, d_model)
            
            # 将历史信息和预测天气都作为Key-Value
            # 这样未来查询可以同时关注历史模式和预测天气
            kv = torch.cat([memory, f_enc], dim=1)  # (B, past_hours + future_hours, d_model)
        else:
            # 只有历史信息作为Key-Value
            kv = memory  # (B, past_hours, d_model)

        # 解码器处理
        # 注意：TransformerDecoder的正确参数顺序是 (tgt, memory)
        # tgt: 目标序列 (Query) - future_queries
        # memory: 编码器输出 (Key-Value) - 历史信息 + 预测天气
        decoded = self.decoder(future_queries, kv)  # (B, future_hours, d_model)

        # 输出预测
        result = self.output_head(decoded).squeeze(-1)  # (B, future_hours)
        
        return result * 100  # 乘以100转换为百分比

# 为了向后兼容，保留原始Transformer类
class Transformer(nn.Module):
    """原始PV预测Transformer模型 - 保持向后兼容"""
    def __init__(
        self,
        hist_dim: int,
        fcst_dim: int,
        config: dict
    ):
        super().__init__()
        self.cfg = config
        d_model = config['d_model']

        # 投影层
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.fcst_proj = nn.Linear(fcst_dim, d_model) if fcst_dim > 0 else None

        # 位置编码
        self.pos_enc = LocalPositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['num_heads'],
            dim_feedforward=d_model * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # 输出头
        self.head = nn.Sequential(
            nn.Linear(d_model, config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['future_hours']),
            nn.Sigmoid()
        )

    def forward(
        self,
        hist: torch.Tensor,        # shape: (B, past_hours, hist_dim)
        fcst: torch.Tensor = None  # shape: (B, future_hours, fcst_dim), optional
    ) -> torch.Tensor:
        # 编码历史输入
        h = self.hist_proj(hist)              # (B, past_hours, d_model)
        h = self.pos_enc(h)
        h_enc = self.encoder(h)               # (B, past_hours, d_model)

        # 编码预测输入（如果适用）
        if self.cfg.get('use_forecast', False) and fcst is not None and self.fcst_proj is not None:
            f = self.fcst_proj(fcst)          # (B, future_hours, d_model)
            f = self.pos_enc(f)
            f_enc = self.encoder(f)           # (B, future_hours, d_model)
            
            # 使用历史编码的最后部分和预测编码
            combined = torch.cat([h_enc, f_enc], dim=1)
        else:
            combined = h_enc

        # 使用最后的时间步进行预测
        last_timestep = combined[:, -1, :]  # (B, d_model)
        result = self.head(last_timestep)   # (B, future_hours)
        
        return result * 100  # 乘以100转换为百分比
