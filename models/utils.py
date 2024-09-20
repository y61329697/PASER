import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ParameterList, Parameter
from typing import List
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        # Global average pooling layer, performs global pooling on the time dimension of the input feature map. Parameter 1 represents the number of output time steps.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Two fully connected layers to generate weights for each channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Adjust input tensor dimensions from (B, T, C) to (B, C, T)
        x = x.permute(0, 2, 1)
        # Get the batch size and number of channels of the input feature map
        b, c, _ = x.size()
        # Global average pooling, averaging the features of each channel, output size is (batch_size, channel)
        y = self.avg_pool(x).view(b, c)
        # Generate channel recalibration coefficients through fully connected layers, output size is (batch_size, channel, 1)
        y = self.fc(y).view(b, c, 1)
        # Weight the input feature map channels using the recalibration coefficients
        y = x * y.expand_as(x)
        # Convert tensor dimensions back to (B, T, C)
        return y.permute(0, 2, 1)
    

class ScalarMix(nn.Module):
    """
    from https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    In addition, if `do_layer_norm=True` then apply layer normalization to each tensor
    before weighting.
    """
    def __init__(
        self,
        mixture_size: int,
        initial_scalar_parameters: List[float] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.mixture_size = mixture_size

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size

        assert len(initial_scalar_parameters
                   ) == self.mixture_size, "{} tensors were passed, but the module was initialized to mix {} tensors.".format(
                       len(initial_scalar_parameters), self.mixture_size)

        self.scalar_parameters = ParameterList(
            [Parameter(torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=trainable) for i in range(mixture_size)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute a weighted average of the `tensors`.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        When `do_layer_norm=True`, the `mask` is required input.  If the `tensors` are
        dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the `mask` is dimensioned
        `(dim_0, ..., dim_{n-1})`, as in the typical case with `tensors` of shape
        `(batch_size, timesteps, dim)` and `mask` of shape `(batch_size, timesteps)`.
        When `do_layer_norm=False` the `mask` is ignored.
        """
        assert len(tensors) == self.mixture_size, "{} tensors were passed, but the module was initialized to mix {} tensors.".format(
            len(tensors), self.mixture_size)

        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for parameter in self.scalar_parameters]), dim=0)
        # 将张量normed_weights 拆分为张量列表，每个张量包含一个元素。
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atten_weight = nn.Parameter(torch.Tensor(hidden_dim, 1), requires_grad=True)
        nn.init.uniform_(self.atten_weight)

    def compute_mask(self, inputs, mask):
        new_attn_mask = torch.zeros_like(mask, dtype=inputs.dtype)
        new_attn_mask.masked_fill_(mask, float("-inf"))  # mask是True

        return new_attn_mask

    def forward(self, inputs, mask=None):

        eij = torch.matmul(inputs, self.atten_weight).squeeze(-1)
        eij = torch.tanh(eij)

        if mask is not None:
            mask = ~mask
            tmask = self.compute_mask(inputs, mask)
            a = torch.softmax(eij + tmask, dim=1).unsqueeze(-1)
        else:
            a = torch.softmax(eij, dim=1).unsqueeze(-1)
        weighted_output = inputs * a
        return weighted_output.sum(dim=1)
    

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        for module in [self.query, self.key, self.value]:
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        
        q = self.query(x[:, 0, :])  # 使用第一个时间步作为查询
        k = self.key(x)
        v = self.value(x)

        attention_scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)

        if mask is not None:
            attention_weights = attention_weights * mask.float()
        
        output = torch.bmm(attention_weights.unsqueeze(1), v).squeeze(1)
        return output
   

class SoftAttentionwithLN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atten_weight = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.atten_weight)

    def compute_mask(self, inputs, mask):
        new_attn_mask = torch.zeros_like(mask, dtype=inputs.dtype)
        new_attn_mask.masked_fill_(mask, float("-inf"))
        return new_attn_mask

    def forward(self, inputs, mask=None):
        inputs = self.layer_norm(inputs)
        eij = torch.matmul(inputs, self.atten_weight).squeeze(-1)
        eij = F.relu(eij)  # 使用ReLU替代tanh

        if mask is not None:
            mask = ~mask
            tmask = self.compute_mask(inputs, mask)
            a = F.softmax(eij + tmask, dim=1).unsqueeze(-1)
        else:
            a = F.softmax(eij, dim=1).unsqueeze(-1)

        weighted_output = inputs * a
        return weighted_output.sum(dim=1)
    

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=768, num_heads=4, dropout=0.1):
        super(AttentionFusion, self).__init__()

        self.num_heads = num_heads
        self.feature_dim = feature_dim

        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_heads)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim) 
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, speech_features, phoneme_features):
        B, T1, D = speech_features.size()
        _, T2, _ = phoneme_features.size()

        attn_weights = []
        for head in self.attention_heads:
            head_attn_weights = []

            for t in range(T1):
                speech_t = speech_features[:, t, :].unsqueeze(1)  # (B, 1, D)
                combined = torch.cat([speech_t.repeat(1, T2, 1), phoneme_features], dim=-1)  # (B, T2, 2*feature_dim)
                attn_weight = head(combined).squeeze(-1)  # (B, T2)
                head_attn_weights.append(attn_weight)
                
            head_attn_weights = torch.stack(head_attn_weights, dim=1)  # (B, T1, T2)
            attn_weights.append(head_attn_weights)

        attn_weights = torch.stack(attn_weights, dim=0)  # (num_heads, B, T1, T2)
        attn_weights = F.softmax(attn_weights, dim=-1).mean(dim=0)  # 平均多头的注意力权重 (B, T1, T2)
        context = torch.bmm(attn_weights, phoneme_features)  # (B, T1, D)
        fused = torch.cat([speech_features, context], dim=-1)  # (B, T1, 2D)
        output = self.fusion(self.dropout(fused))  # (B, T1, D)
        output = output.mean(dim=1)  # (B, D)

        return output


class MultimodalTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=1):
        super().__init__()
        self.speech_encoder = nn.Linear(d_model, d_model)
        self.phoneme_encoder = nn.Linear(d_model, d_model)

        # add a learnable separator
        self.separator = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def forward(self, speech_feat, phoneme_feat):
        speech_encoded = self.speech_encoder(speech_feat)
        phoneme_encoded = self.phoneme_encoder(phoneme_feat)

        batch_size = speech_feat.size(0)
        separator = self.separator.expand(batch_size, -1, -1)
        fused_feat = torch.cat([speech_encoded, separator, phoneme_encoded], dim=1)

        transformer_out = self.transformer(fused_feat)
        pooled_feat = transformer_out.mean(dim=1)
        return pooled_feat
    
    
def get_padding_mask(max_len, batch_size, lengths):
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    # x implicitly has a leading dimension of 1, so the 0th dimension can be expanded using `expand`
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths[:, None]
    return mask

def entropy_loss(output):
    # Apply softmax to the output
    softmax_output = F.softmax(output, dim=1)
    # Calculate entropy
    entropy = -torch.mean(torch.sum(softmax_output * torch.log(softmax_output + 1e-10), dim=1))
    return entropy



