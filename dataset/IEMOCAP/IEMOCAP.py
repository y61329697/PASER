from regex import P
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Config, Wav2Vec2Model
from torch.utils.data import Sampler
from collections import defaultdict
import random
import numpy as np

config = Wav2Vec2Config()
model = Wav2Vec2Model(config)

# 五折划分数据集
def Partition(ith_fold, DataMap, batch_size):
    '''
    Args:
        ith_fold: the test session for iemocap, ranging from 1 to 5
    '''
    train_list = []
    test_list = DataMap['Session{}'.format(ith_fold)]
    for i in range(1, 6):
        sess_name = 'Session{}'.format(i)
        if i != ith_fold:
            train_list.extend(DataMap[sess_name])
    train_list = list(filter(lambda x: x['audio_length'] < 300000, train_list)) # For training efficiency
    train_dataset = IEMOCAP(train_list)
    test_dataset = IEMOCAP(test_list) 
    train_sampler = DiverseSpeakerEmoBatchSampler(train_list, batch_size=batch_size)
    
    return train_dataset, test_dataset, train_sampler

def Merge(DataMap):
    train_list = []
    for i in range(1, 6):
        sess_name = 'Session{}'.format(i)
        train_list.extend(DataMap[sess_name])
    train_list = list(filter(lambda x: x['audio_length'] < 300000, train_list)) # For training efficiency
    train_dataset = IEMOCAP(train_list)
    return train_dataset

def get_feat_extract_output_lengths(input_length):
    """
        Computes the output length of the convolutional layers
        """
    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        input_length = _conv_out_length(input_length, kernel_size, stride)
    return input_length


class IEMOCAP(data.Dataset):
    """Speech dataset."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['audio'], self.data[index]['audio_length'], self.data[index]['tlabel'], self.data[index][
            'label'], self.data[index]['text'], self.data[index]['filename'], self.data[index]['plabel']


    def collate_fn(self, datas):
        ''' Padding audio and tlabel dynamically, computing length of pretrained features
        Args:
            datas: List[(index: int, audio: torch.Tensor, audio_length: int, tlabel: List[int], label: int, plabel: torch.Tensor, text: str)]
        '''
        # 从可以容纳不同长度的 List 转换为 Tensor
        audio = [data[0] for data in datas]
        padded_audio = pad_sequence(audio, batch_first=True, padding_value=0)
        audio_length = torch.tensor([get_feat_extract_output_lengths(data[1]) for data in datas])
        tlabel = [torch.tensor(data[2]) for data in datas]
        tlabel_length = torch.tensor([tokens.size(0) for tokens in tlabel])
        padded_tlabel = pad_sequence(tlabel, batch_first=True, padding_value=-1)
        label = torch.tensor([data[3] for data in datas])
        text = [data[4] for data in datas]
        filename = [data[5] for data in datas]
        plabel = [torch.tensor(data[6]) for data in datas]
        ptable_length = torch.tensor([tokens.size(0) for tokens in plabel])
        padded_plabel = pad_sequence(plabel, batch_first=True, padding_value=0)

        return padded_audio, audio_length, padded_tlabel, tlabel_length, label, \
                text, filename, padded_plabel, ptable_length
                
class DiverseSpeakerBatchSampler(Sampler):
    def __init__(self, data, batch_size, seed=42):
        self.batch_size = batch_size
        self.data = data
        self.seed = seed
        self.spk_dict = {'Ses01M': 0, 'Ses01F': 1, 'Ses02M': 2, 'Ses02F': 3, 'Ses03M': 4, 'Ses03F': 5,
                         'Ses04M': 6, 'Ses04F': 7, 'Ses05M': 8, 'Ses05F': 9}
        self.speaker_to_indices = self._group_speakers(data)
        random.seed(self.seed)

    def _group_speakers(self, data):

        speaker_to_indices = defaultdict(list)
        for idx, entry in enumerate(data):
            try:
                speaker_id = self.spk_dict[entry['filename'].split('_')[0]]
                speaker_to_indices[speaker_id].append(idx)
            except KeyError:
                print(f"警告：文件名 {entry['filename']} 的格式不正确或说话人ID未知")
        return speaker_to_indices

    def __iter__(self):
        speakers = list(self.speaker_to_indices.keys())
        batches = []
        
        for speaker_id in speakers:
            random.shuffle(self.speaker_to_indices[speaker_id])

        remaining_indices = {k: v.copy() for k, v in self.speaker_to_indices.items()}

        while any(remaining_indices.values()):
            batch = []

            while len(batch) < self.batch_size and any(remaining_indices.values()):
                random.shuffle(speakers)
                for speaker in speakers:
                    if remaining_indices[speaker]:
                        batch.append(remaining_indices[speaker].pop(0))
                        if len(batch) == self.batch_size:
                            break
            
            if batch:  
                batches.append(batch)

        index = [idx for batch in batches for idx in batch]
        return iter(index)

    def __len__(self):
        return sum(len(indices) for indices in self.speaker_to_indices.values())


def any_remaining(remaining_indices):
    return any(indices for speakers in remaining_indices.values() for indices in speakers.values())

def has_empty_list(remaining_indices):
    for speakers in remaining_indices.values():
        for indices in speakers.values():
            if not indices:  # 列表为空
                return True
    return False
    
class DiverseSpeakerEmoBatchSampler(Sampler):
    def __init__(self, data, batch_size, seed=42):
        self.batch_size = batch_size
        self.data = data
        self.seed = seed
        self.spk_dict = {'Ses01M': 0, 'Ses01F': 1, 'Ses02M': 2, 'Ses02F': 3, 'Ses03M': 4, 'Ses03F': 5,
                         'Ses04M': 6, 'Ses04F': 7, 'Ses05M': 8, 'Ses05F': 9}
        self.speaker_emo_to_indices = self._group_speakers_emo(data)
        random.seed(self.seed)

    def _group_speakers_emo(self, data):

        speaker_emo_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, entry in enumerate(data):
            try:
                speaker_id = self.spk_dict[entry['filename'].split('_')[0]]
                emo_id = entry['label']
                speaker_emo_to_indices[emo_id][speaker_id].append(idx)
            except KeyError:
                print(f"警告：文件名 {entry['filename']} 的格式不正确或说话人ID未知")
        return speaker_emo_to_indices

    def __iter__(self):
        emos = list(self.speaker_emo_to_indices.keys())
        speakers = list(self.speaker_emo_to_indices[0].keys())
        batches = []
        
        for emo in emos:
            for speaker_id in speakers:
                random.shuffle(self.speaker_emo_to_indices[emo][speaker_id])

        remaining_indices = {emo: {speaker: indices.copy() for speaker, indices in speakers.items()} 
                     for emo, speakers in self.speaker_emo_to_indices.items()}

        while any_remaining(remaining_indices):
            
            batch = []
            if has_empty_list(remaining_indices):
                break
            
            while len(batch) < self.batch_size and any_remaining(remaining_indices):
                random.shuffle(speakers)
                random.shuffle(emos)
                per_emo_spk = int(len(speakers) / len(emos))
                for idx, emo in enumerate(emos):
                    for speaker in speakers[idx*per_emo_spk: (idx+1)*per_emo_spk]:
                        if remaining_indices[emo][speaker]:
                            batch.append(remaining_indices[emo][speaker].pop(0))
                            if len(batch) == self.batch_size:
                                break
            
            if batch:
                batches.append(batch)
    
        if any_remaining(remaining_indices):
            
            remaining_indices_1D = defaultdict(list)
            
            for emo, speakers in remaining_indices.items():
                for indices in speakers.values():
                    remaining_indices_1D[emo].extend(indices)
                    
            while any(remaining_indices_1D.values()):
                batch = []
                
                while len(batch) < self.batch_size and any(remaining_indices_1D.values()):
                    random.shuffle(emos)
                    for emo in emos:
                        if remaining_indices_1D[emo]:
                            batch.append(remaining_indices_1D[emo].pop(0))
                            if len(batch) == self.batch_size:
                                break
                
                if batch:
                    batches.append(batch)

        index = [idx for batch in batches for idx in batch]
        return iter(index)

    def __len__(self):
        return sum(len(indices) for indices in self.speaker_emo_to_indices.values())
                

