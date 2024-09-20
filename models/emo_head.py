import torch
import torch.nn as nn
from .utils import (get_padding_mask, SoftAttention, SELayer1D, SoftAttentionwithLN,
                    TemporalAttention, AttentionFusion, MultimodalTransformer)
from .grl import ReverseLayerF
from .feature_fusion import project_algorithm
from .cross_fuse import CrossAttentionWithFeedForward, CrossAttention

class Emo_Head(nn.Module):
    def __init__(self, enc_dim, mode='mean', fuse_mode='concat', emo_num=4, speaker_num=10, all_args=None):
        super().__init__()
        self.mode = mode
        self.ablation_level = all_args.ablation_level
        self.fuse_mode = fuse_mode
        self.half_enc_dim = int (0.5 * enc_dim) 
        
        self.att_audio = SoftAttentionwithLN(self.half_enc_dim)
        self.att_phoneme = SoftAttentionwithLN(self.half_enc_dim)

        self.se_audio = SELayer1D(channel=self.half_enc_dim)
        self.se_phoneme = SELayer1D(channel=self.half_enc_dim)
        
        self.audio_head = nn.Linear(self.half_enc_dim, emo_num)
        self.phoneme_head = nn.Linear(self.half_enc_dim, emo_num)
        
        self.ablation_emo_head = nn.Linear(self.half_enc_dim, emo_num)
        self.ablation_norm = nn.LayerNorm(self.half_enc_dim)
        
        self.norm_spk = nn.LayerNorm(self.half_enc_dim)
        self.norm_audio = nn.LayerNorm(self.half_enc_dim)
        self.norm_phoneme = nn.LayerNorm(self.half_enc_dim)
        self.norm_gender = nn.LayerNorm(self.half_enc_dim)
        self.norm_project = nn.LayerNorm(self.half_enc_dim)
        
        self.t_att1 = TemporalAttention(self.half_enc_dim)
        self.t_att2 = TemporalAttention(self.half_enc_dim)
        self.att_fusion = AttentionFusion(feature_dim=self.half_enc_dim, num_heads=2)

        if fuse_mode == 'bi_linear':
            self.bi_linear = nn.Bilinear(self.half_enc_dim, self.half_enc_dim, self.half_enc_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.half_enc_dim * 2, self.half_enc_dim),
            nn.Sigmoid()
        )
        self.multi_transformer = MultimodalTransformer(d_model=self.half_enc_dim)

        self.gender_head_full = nn.Linear(enc_dim, emo_num)
        self.norm_gender_full = nn.LayerNorm(enc_dim)
        
        self.gender_head = nn.Linear(self.half_enc_dim, 2)
        self.spk_head = nn.Linear(self.half_enc_dim, speaker_num)
        
        if self.fuse_mode == 'cross' or self.fuse_mode == 'cross_simple':
            self.cross1 = CrossAttentionWithFeedForward(d_model=self.half_enc_dim, 
                                                        nhead=12, d_ff=self.half_enc_dim)
            self.cross2 = CrossAttentionWithFeedForward(d_model=self.half_enc_dim, 
                                                        nhead=12, d_ff=self.half_enc_dim)
            self.cross1_without_ffn = CrossAttention(d_model=self.half_enc_dim, nhead=12)
            self.cross2_without_ffn = CrossAttention(d_model=self.half_enc_dim, nhead=12)
            
        if self.fuse_mode == 'project' or self.mode == 'att_fusion' or self.fuse_mode == 'gate' or self.fuse_mode == 'bi_linear':
            enc_dim = self.half_enc_dim

        self.head = nn.Linear(enc_dim, emo_num)
        self.project_emo_head = nn.Linear(enc_dim, emo_num)
        
        self.norm = nn.LayerNorm(enc_dim)
        self._initialize_weights()

    def forward(self, hidden_states, audio_length, phoneme_states, phoneme_length, alpha):

        B, T, E = hidden_states.shape
        B_, T_, E_ = phoneme_states.shape
        if self.mode == 'mean':
            att_mask = get_padding_mask(T, B, audio_length).unsqueeze(-1).expand(-1, -1, E).bool()
            hidden_states = hidden_states.masked_fill(~att_mask, 0)
            
            # Ablation experiment requires commenting out the squeeze-and-excitation module
            if self.ablation_level == -1 or self.ablation_level == 1:
                hidden_states = self.se_audio(hidden_states)

            if self.fuse_mode == 'cross':
                a2p_features, _ = self.cross1(hidden_states, phoneme_states, phoneme_states)
                a2p_features = torch.sum(a2p_features, dim=1) / audio_length.unsqueeze(-1)
            elif self.fuse_mode == 'cross_simple':
                a2p_features, _ = self.cross1_without_ffn(hidden_states, phoneme_states, phoneme_states)
                a2p_features = torch.sum(a2p_features, dim=1) / audio_length.unsqueeze(-1)
                
            
            att_mask_p = get_padding_mask(T_, B_, phoneme_length).unsqueeze(-1).expand(-1, -1, E_).bool()
            phoneme_states = phoneme_states.masked_fill(~att_mask_p, 0)
            if self.ablation_level == -1 or self.ablation_level == 1:
                phoneme_states = self.se_phoneme(phoneme_states)

            if self.fuse_mode == 'cross':
                p2a_features, _ = self.cross2(phoneme_states, hidden_states, hidden_states)
                p2a_features = torch.sum(p2a_features, dim=1) / phoneme_length.unsqueeze(-1)
            elif self.fuse_mode == 'cross_simple':
                p2a_features, _ = self.cross2_without_ffn(phoneme_states, hidden_states, hidden_states)
                p2a_features = torch.sum(p2a_features, dim=1) / phoneme_length.unsqueeze(-1)
                
            hidden_states = torch.sum(hidden_states, dim=1) / audio_length.unsqueeze(-1)
            phoneme_states = torch.sum(phoneme_states, dim=1) / phoneme_length.unsqueeze(-1)
            
            if self.fuse_mode == 'cross' or self.fuse_mode == 'cross_simple':
                hidden_states = a2p_features
                phoneme_states = p2a_features
            
        elif self.mode == 'att':
            att_mask = get_padding_mask(T, B, audio_length).unsqueeze(-1).expand(-1, -1, E).bool()
            hidden_states = hidden_states.masked_fill(~att_mask, 0)
            # Ablation experiment requires commenting out the squeeze-and-excitation module
            if self.ablation_level == -1 or self.ablation_level == 1:
                hidden_states = self.se_audio(hidden_states)
            att_padding_mask = get_padding_mask(T, B, audio_length).bool()
            
            if self.fuse_mode == 'cross':
                a2p_features, _ = self.cross1(hidden_states, phoneme_states, phoneme_states)
                a2p_features = self.att_audio(a2p_features, att_padding_mask.bool())
            elif self.fuse_mode == 'concat' or self.fuse_mode =='project':   
                hidden_states = self.att_audio(hidden_states, att_padding_mask.bool())
            elif self.fuse_mode == 'cross_simple':
                a2p_features, _ = self.cross1_without_ffn(hidden_states, phoneme_states, phoneme_states)
                a2p_features = torch.sum(a2p_features, dim=1) / audio_length.unsqueeze(-1)
            
            att_mask_p = get_padding_mask(T_, B_, phoneme_length).unsqueeze(-1).expand(-1, -1, E_).bool()
            phoneme_states = phoneme_states.masked_fill(~att_mask_p, 0)
            # Ablation experiment requires commenting out the squeeze-and-excitation module
            if self.ablation_level == -1 or self.ablation_level == 1:
                phoneme_states = self.se_phoneme(phoneme_states)
            att_padding_mask_p = get_padding_mask(T_, B_, phoneme_length)
            if self.fuse_mode == 'cross':
                p2a_features, _ = self.cross2(phoneme_states, hidden_states, hidden_states)
                p2a_features = self.att_phoneme(p2a_features, att_padding_mask_p.bool())
            elif self.fuse_mode == 'concat' or self.fuse_mode =='project':   
                phoneme_states = self.att_phoneme(phoneme_states, att_padding_mask_p.bool())
            elif self.fuse_mode == 'cross_simple':
                p2a_features, _ = self.cross2_without_ffn(phoneme_states, hidden_states, hidden_states)
                p2a_features = torch.sum(p2a_features, dim=1) / phoneme_length.unsqueeze(-1)
                
            if self.fuse_mode == 'cross' or self.fuse_mode == 'cross_simple':
                hidden_states = a2p_features
                phoneme_states = p2a_features
                
        elif self.mode == 't_att':
            att_mask = get_padding_mask(T, B, audio_length)
            hidden_states = self.se_audio(hidden_states)
            hidden_states = self.t_att1(hidden_states, att_mask)
            
            att_mask_p = get_padding_mask(T_, B_, phoneme_length)
            phoneme_states = self.se_phoneme(phoneme_states)
            phoneme_states = self.t_att2(phoneme_states)

        elif self.mode == 'att_fusion':
            self.fuse_mode = 'null'
            att_mask = get_padding_mask(T, B, audio_length).unsqueeze(-1).expand(-1, -1, E).bool()
            hidden_states = self.se_audio(hidden_states)
            hidden_states = hidden_states.masked_fill(~att_mask, 0)

            att_mask_p = get_padding_mask(T_, B_, phoneme_length).unsqueeze(-1).expand(-1, -1, E_).bool()
            phoneme_states = self.se_phoneme(phoneme_states)
            phoneme_states = phoneme_states.masked_fill(~att_mask_p, 0)
            fusion_features = self.att_fusion(hidden_states, phoneme_states)

            hidden_states = torch.sum(hidden_states, dim=1) / audio_length.unsqueeze(-1)
            phoneme_states = torch.sum(phoneme_states, dim=1) / phoneme_length.unsqueeze(-1)

        elif self.mode == 'multi_trans':

            self.fuse_mode = 'null'
            att_mask = get_padding_mask(T, B, audio_length).unsqueeze(-1).expand(-1, -1, E).bool()
            hidden_states = self.se_audio(hidden_states)
            hidden_states = hidden_states.masked_fill(~att_mask, 0)

            att_mask_p = get_padding_mask(T_, B_, phoneme_length).unsqueeze(-1).expand(-1, -1, E_).bool()
            phoneme_states = self.se_phoneme(phoneme_states)
            phoneme_states = phoneme_states.masked_fill(~att_mask_p, 0)
            fusion_features = self.multi_transformer(hidden_states, phoneme_states)
            hidden_states = torch.sum(hidden_states, dim=1) / audio_length.unsqueeze(-1)
            phoneme_states = torch.sum(phoneme_states, dim=1) / phoneme_length.unsqueeze(-1)

        if self.fuse_mode == 'concat' or self.fuse_mode == 'cross' or self.fuse_mode == 'cross_simple':

            features = torch.cat([hidden_states, phoneme_states], dim=-1)
            reverse_features_hid = ReverseLayerF.apply(hidden_states, alpha)
            out = self.head(self.norm(features))
            out2 = self.spk_head(self.norm_spk(reverse_features_hid))
            
        elif self.fuse_mode == 'project':
            # Gradient reversal layer must be used when using projection
            # The purpose is to make phoneme features not discriminative
            features = project_algorithm(hidden_states, phoneme_states)
            reserve_phoneme_features = ReverseLayerF.apply(phoneme_states, alpha)
            
            out = self.head(self.norm(features))
            out2 = self.project_emo_head(self.norm_project(reserve_phoneme_features))

        elif self.fuse_mode == 'gate':
            combined = torch.cat([hidden_states, phoneme_states], dim=-1)  # (B, 2*D)
            gate = self.gate(combined)  # (B, D)
            fused = gate * self.norm_audio(hidden_states) + (1 - gate) * self.norm_phoneme(phoneme_states)  # (B, D)
            out = self.head(fused)

            reverse_features_hid = ReverseLayerF.apply(hidden_states, alpha)
            out2 = self.spk_head(self.norm_spk(reverse_features_hid))

        elif self.fuse_mode == 'null':
            out = self.head(self.norm(fusion_features))

            reverse_features_hid = ReverseLayerF.apply(hidden_states, alpha)
            out2 = self.spk_head(self.norm_spk(reverse_features_hid))

        elif self.fuse_mode == 'bi_linear':
            # Bilinear pooling scheme
            bi_features = self.bi_linear(self.norm_audio(hidden_states), self.norm_phoneme(phoneme_states))
            out = self.head(self.norm(bi_features))

            reverse_features_hid = ReverseLayerF.apply(hidden_states, alpha)
            out2 = self.spk_head(self.norm_spk(reverse_features_hid))

            
        audio_logits = self.audio_head(self.norm_audio(hidden_states))
        phoneme_logits = self.phoneme_head(self.norm_phoneme(phoneme_states))
        gender_logits = self.gender_head_full(self.norm_gender_full(torch.cat([hidden_states, phoneme_states], dim=-1)))
        
        # Ablation experiment, not concatenating with phoneme labels
        if self.ablation_level == 0 or self.ablation_level == 1:
            out = self.ablation_emo_head(self.ablation_norm(hidden_states))
        
        return out, out2, hidden_states, phoneme_states, audio_logits, phoneme_logits, gender_logits
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
