from sympy import im
from .pretrain import Speech_Pretrain_Model
from .emo_head import Emo_Head
from .paser import PASER
from .phoneme_decoder import PhonemeDecoder


def build_model(hid_dim=768, pretrain='wav2vec2', finetune=True, mode='att', fuse_mode='concat', emo_num=4, 
                phoneme_weight=1.0, spk_weight=0.1, otherhead_weight=0.2, all_args=None):

    auto_enc = None
    pretrain = Speech_Pretrain_Model(pretrain, finetune)
    emo_head = Emo_Head(enc_dim=hid_dim * 2, mode=mode, fuse_mode=fuse_mode, emo_num=emo_num, all_args=all_args)
    phoneme_decoder = PhonemeDecoder(phoneme_vocab_size=75, d_model=768, num_decoder_layers=2,
                                     nhead=12, dim_feedforward=768, dropout=0.1)

    emo_cls = PASER(pretrain, emo_head, phoneme_decoder,
                phoneme_weight=phoneme_weight, spk_weight=spk_weight, otherhead_weight=otherhead_weight, all_args=all_args)
    return emo_cls

