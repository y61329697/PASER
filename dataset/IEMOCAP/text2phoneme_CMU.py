import nltk
import csv
from nltk.corpus import cmudict
from collections import OrderedDict

cmu_dict = cmudict.dict()

def generate_cmudict2id(save_path, reserve_token: list):
    # nltk.download('cmudict')
    cmu_dict = nltk.corpus.cmudict.dict()
    phoneme_dict = OrderedDict()
    
    for i, token in enumerate(reserve_token):
        phoneme_dict[token] = i
    
    initial_index = len(reserve_token)
    
    for word, phoneme_seqs in cmu_dict.items():
        for phoneme_seq in phoneme_seqs:
            for phoneme in phoneme_seq:
                phoneme_cleaned = phoneme
                if phoneme_cleaned not in phoneme_dict:
                    phoneme_dict[phoneme_cleaned] = initial_index
                    initial_index += 1

    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Phoneme', 'ID'])
        for phoneme, idx in phoneme_dict.items():
            writer.writerow([phoneme, idx])
