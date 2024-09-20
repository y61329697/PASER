import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from transformers import Wav2Vec2Model, Wav2Vec2Config, HubertModel, HubertConfig, WavLMModel, WavLMConfig, AutoModel
from .utils import ScalarMix

class Speech_Pretrain_Model(nn.Module):

    def __init__(self, pretrain='wav2vec2', finetune=False) -> None:
        super().__init__()
        self.finetune = finetune
        configmap = {'mask_time_prob': 0.08, 'mask_time_length': 15, 'mask_feature_prob': 0.05, 'mask_feature_length': 64}
        assert pretrain in ['hubert', 'wav2vec2', 'wavlm'], "Unkown pretrain model for finetuning"

        self.epoch = 0

        if pretrain == 'hubert':
            config = HubertConfig.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=~finetune)
            self.pretrain = HubertModel.from_pretrained("facebook/hubert-base-ls960", config=config)
        elif pretrain == 'wav2vec2':
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base", output_hidden_states=~finetune)

            # don't mask
            config.mask_time_prob = 0.0
            config.mask_feature_prob = 0.0
            
            self.pretrain = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", config=config)
        elif pretrain == 'wavlm':
            config = WavLMConfig.from_pretrained("microsoft/wavlm-base", output_hidden_states=~finetune)
            self.pretrain = WavLMModel.from_pretrained("microsoft/wavlm-base", config=config)

        if finetune:
            # only freeze the feature extractor(conv1d layers), transformer layers are trainable
            if pretrain == 'wav2vec2':
                self.pretrain.freeze_feature_encoder() 
            if pretrain == 'hubert':
                self.pretrain.feature_extractor._freeze_parameters()
            self.freeze_epoch = 0
        else:
            self.weight_audio = ScalarMix(13)
    
    def forward(self, x):

        if self.finetune:
            ft = self.finetune
            # only use the last layer output
            with torch.no_grad() if not ft or not self.training else contextlib.ExitStack():
                x_audio = self.pretrain(x).last_hidden_state
                
        else:
            self.pretrain.eval()
            with torch.no_grad():
                x = self.pretrain(x).hidden_states

            x_audio = self.weight_audio(x)

        return x_audio

