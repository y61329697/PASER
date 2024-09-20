import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List
from .utils import get_padding_mask, entropy_loss
from .cross_fuse import CrossAttentionWithFeedForward
import numpy as np

spk_dict = {'Ses01M': 0, 'Ses01F': 1, 'Ses02M': 2, 'Ses02F': 3, 'Ses03M': 4, 'Ses03F': 5,
            'Ses04M': 6, 'Ses04F': 7, 'Ses05M': 8, 'Ses05F': 9}
gender_dict = {'M': 0, 'F': 1}

@dataclass
class ClsOutput(ModelOutput):
    loss: torch.FloatTensor = None
    loss_spk: torch.FloatTensor = None
    loss_emo: Optional[torch.FloatTensor] = None
    loss_phoneme: Optional[torch.FloatTensor] = None
    loss_domain:  Optional[torch.FloatTensor] = None
    loss_audio_emo: Optional[torch.FloatTensor] = None
    loss_phoneme_emo: Optional[torch.FloatTensor] = None
    loss_gender: Optional[torch.FloatTensor] = None
    head_logits: Optional[torch.FloatTensor] = None
    phoneme_logits: Optional[torch.FloatTensor] = None
    gender_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    phoneme_states: Optional[torch.FloatTensor] = None
    losses: Optional[List[torch.FloatTensor]] = None

class PASER(nn.Module):
    def __init__(self, pretrain, emo_head, phoneme_decoder, 
                 phoneme_weight=1.0, spk_weight=0.1, otherhead_weight=0.2, all_args=None):
        super().__init__()
        
        self.pretrain = pretrain
        self.emo_head = emo_head
        self.phoneme_dec = phoneme_decoder
        
        self.phoneme_weight = phoneme_weight
        self.spk_weight = spk_weight
        self.otherhead_weight = otherhead_weight
        self.ablation_level = all_args.ablation_level
        
        
    def forward(self, audio, audio_length, labels, plabels, plable_length, filenames, now_step, sum_step):

        hidden_states = self.pretrain(audio)
        
        B_audio, T_audio, E_audio = hidden_states.shape
        audio_padding_mask = get_padding_mask(T_audio, B_audio, audio_length)
        
        plabels_input = plabels[:, :-1]
        plabels_target = plabels[:,1:]
        
        p = float(now_step / sum_step)
        alpha = torch.tensor(2. / (1. + np.exp(-10 * p)) - 1)
        
        if self.phoneme_weight > 0.0 and (self.ablation_level == -1 or self.ablation_level == 2):
            P_logits, P_states = self.phoneme_dec(plabels_input, hidden_states, memory_key_padding_mask=~audio_padding_mask)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            phoneme_loss = criterion(P_logits.reshape(-1, P_logits.shape[-1]), plabels_target.reshape(-1))
        else:
            if self.ablation_level == 0 or self.ablation_level == 1:
            # Placeholder for ablation experiments
                P_states = torch.zeros(size=(B_audio, 10, 768)).cuda()
                P_logits = torch.zeros(size=(B_audio, 10, 75)).cuda()
            phoneme_loss = torch.tensor(0.0)
        
        emo_logits, sec_logits, raw_hidden_states, phoneme_states, audio_logits, phoneme_logits, gender_logits= \
        self.emo_head(hidden_states, audio_length, P_states, plable_length, alpha)
        id_labels, gender_labels = [], []
        for filename in filenames:
            id_labels.append(spk_dict[filename.split('_')[0]])
            gender_labels.append(gender_dict[filename.split('_')[0][-1]])

        id_labels = torch.tensor(id_labels).cuda()
        gender_labels = torch.tensor(gender_labels).cuda()
        class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).cuda() 
           
        loss_emo = F.cross_entropy(emo_logits, labels, weight=class_weights)
        
        # The following losses are not used in training, mainly for observation purposes
        loss_audio = F.cross_entropy(audio_logits, labels, weight=class_weights)
        loss_phoneme = F.cross_entropy(phoneme_logits, labels, weight=class_weights)
        loss_gender = F.cross_entropy(gender_logits, gender_labels)
        
        # The following losses are not used in training, mainly for observation purposes
        if self.emo_head.fuse_mode == 'concat' or self.emo_head.fuse_mode == 'cross':
            loss_spk = F.cross_entropy(sec_logits, id_labels)
            loss_domain = entropy_loss(sec_logits)
        elif self.emo_head.fuse_mode == 'project':
            assert self.spk_weight > 0, f"Value must be greater than 0, but got {self.spk_weight}"
            loss_domain = F.cross_entropy(sec_logits, labels)
            loss_spk = torch.tensor(0.0).cuda()
        else:
            loss_domain = torch.tensor(0.0).cuda()
            loss_spk = torch.tensor(0.0).cuda()
            
        return ClsOutput(loss=loss_emo + self.phoneme_weight * phoneme_loss + self.spk_weight * loss_domain \
                         + self.otherhead_weight*(loss_phoneme+loss_audio),
                         loss_spk=self.spk_weight * loss_spk,
                         loss_phoneme = self.phoneme_weight * phoneme_loss,
                         loss_emo=loss_emo, loss_domain = self.spk_weight * loss_domain, 
                         loss_audio_emo=self.otherhead_weight*loss_audio, loss_phoneme_emo=self.otherhead_weight*loss_phoneme,
                         loss_gender=0.0 * loss_gender,
                         head_logits=emo_logits,
                         phoneme_logits=P_logits,
                         gender_logits=gender_logits,
                         hidden_states=raw_hidden_states,
                         phoneme_states=phoneme_states,
                         losses=[loss_emo, phoneme_loss])
