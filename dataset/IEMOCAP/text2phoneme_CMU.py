import nltk
import csv
from nltk.corpus import cmudict
from collections import OrderedDict

# 获取 CMU 发音词典
cmu_dict = cmudict.dict()

# 示例英文单词列表
words = ["hello", "world", "cat", "dog", "house", "computer"]

# 构建字典
word_to_phonemes = {}
for word in words:
    if word in cmu_dict:
        phoneme_seq = cmu_dict[word][0]  # 选择第一个发音, 不同的单词存在多个发音
        word_to_phonemes[word] = phoneme_seq

# 打印结果
for word, phonemes in word_to_phonemes.items():
    print(f"{word}: {phonemes}")

# 输出示例
# hello: ['HH', 'AH0', 'L', 'OW1']
# world: ['W', 'ER1', 'L', 'D']
# cat: ['K', 'AE1', 'T']
# dog: ['D', 'AO1', 'G']
# house: ['HH', 'AW1', 'S']
# computer: ['K', 'AH0', 'M', 'P', 'Y', 'UW1', 'T', 'ER0']

def generate_cmudict2id(save_path, reserve_token: list):
    # 下载 CMU Pronouncing Dictionary，如果尚未下载
    # nltk.download('cmudict')
    
    # 获取 CMU 发音词典
    cmu_dict = nltk.corpus.cmudict.dict()
    
    # 创建有序字典
    phoneme_dict = OrderedDict()
    
    # 添加保留的 token
    for i, token in enumerate(reserve_token):
        phoneme_dict[token] = i
    
    # 获取初始索引
    initial_index = len(reserve_token)
    
    # 添加 CMU 词典中的音素
    for word, phoneme_seqs in cmu_dict.items():
        for phoneme_seq in phoneme_seqs:
            for phoneme in phoneme_seq:
                # phoneme_cleaned = phoneme.strip('0123456789')  # 去掉音素后的数字,不区分重音级别
                phoneme_cleaned = phoneme
                if phoneme_cleaned not in phoneme_dict:
                    phoneme_dict[phoneme_cleaned] = initial_index
                    initial_index += 1
    
    # 保存为 CSV 文件
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Phoneme', 'ID'])  # 写入表头
        for phoneme, idx in phoneme_dict.items():
            writer.writerow([phoneme, idx])

# 使用示例
reserve_token = ['<pad>', '<ukn>', '<bos>', '<eos>', '<blk>']
save_path = 'phoneme_dictcmu.csv'
generate_cmudict2id(save_path, reserve_token)
            