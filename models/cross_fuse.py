import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def forward(self, query, key, value):
        # Transpose to have sequence length as the first dimension
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        
        # Transpose back to have batch size as the first dimension
        attn_output = attn_output.transpose(0, 1)
        output = self.norm(query.transpose(0, 1) + self.dropout(attn_output))
        return output, attn_weights
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.multihead_attn.in_proj_weight)
        nn.init.constant_(self.multihead_attn.in_proj_bias, 0)
        nn.init.xavier_uniform_(self.multihead_attn.out_proj.weight)
        nn.init.constant_(self.multihead_attn.out_proj.bias, 0)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x2 = self.linear2(self.activation(self.linear1(x)))
        x2 = self.dropout(x2)
        output = self.norm(x + x2)
        return output
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.linear2.bias, 0)
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

class CrossAttentionWithFeedForward(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(CrossAttentionWithFeedForward, self).__init__()
        self.cross_attention = CrossAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self._initialize_weights()

    def forward(self, query, key, value):
        attn_output, attn_weights = self.cross_attention(query, key, value)
        output = self.feed_forward(attn_output)
        return output, attn_weights
    
    def _initialize_weights(self):
        pass
