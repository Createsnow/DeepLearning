# -*- coding:utf-8 -*-

import os
import random
import math
import jieba
import codecs
from collections import Counter


def char_mapping(sentences, lower):
    chars = [[s[0].lower() if lower else s[0] for s in sentence] for sentence in sentences]
    # dico,char_to_id, id_to_char=create_dico(chars)
    dico = create_dico(chars)
    dico['<PAD>'] = 1000000001
    dico['<UNK>'] = 1000000000
    char_to_id, id_to_char = create_mapping(dico)
    return dico, char_to_id, id_to_char


def create_dico(sentences):
    assert type(sentences) is list
    # 第一种
    dico = {}
    for sentence in sentences:
        for char in sentence:
            if char not in dico:
                dico[char] = 1
            if char in dico:
                dico[char] += 1
    # 第二种
    # dico=[]
    # char_to_id={}
    # id_to_char={}
    # for sentence in sentences:
    #     dico.extend(sentence)
    # dico=Counter(dico)
    # dico['<PAD>']=100000001
    # dico['<UNK>']=100000000
    # dico=dico.most_common()
    # for id,(char,_) in enumerate(dico):
    #     char_to_id[char]=id
    #     id_to_char[id]=char
    # return dico,char_to_id,id_to_char
    return dico


def create_mapping(dico):
    sorted_char = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_char = {i: k for i, (k, v) in enumerate(sorted_char)}
    char_to_id = {v: k for k, v in id_to_char.items()}
    return char_to_id, id_to_char

def augment_with_pretrained(dictioary, ext_emb_path, char):
    """
       Augment the dictionary with words that have a pretrained embedding.
       If `words` is None, we add every word that has a pretrained embedding
       to the dictionary, otherwise, we only add the words that are given by
       `words` (typically the words in the development and test sets.)
       """
    # print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def tag_mapping(data):
    dico=[[ s[-1] for s in sentence ]for sentence in data]
    tag=create_dico(dico)
    tag_to_id,id_to_tag=create_mapping(tag)
    return dico,tag_to_id,id_to_tag

def prepare_dataset(sentences,char_to_id,tag_to_id,lower=False):
    # chars=[[ char_to_id[s[0]] for s in sentence]for sentence in data]
    # tags=[[ tag_to_id[s[1]] for s in sentence]for sentence in data]
    data=[]
    def f(x):
        return x.lower() if lower else x
    for sentence in sentences:
        string=[s[0] for s in sentence]
        #考虑到未知词的输入计算并对应相对的
        char=[char_to_id[f(x) if f(x) in char_to_id else '<UNK>' ] for x in string]
        seg=get_seg_features(''.join(string))
        tag=[tag_to_id[s[-1]] for s in sentence]
        data.append([string,char,seg,tag])
    return data

def get_seg_features(string):
    """
    0表示一个字
    13表示俩个字组成的词
    123表示三个字组成的词
    1223表示四个字组成的词
    。。。。
    :param string:
    :return:
    """
    seg_feature=[]
    for char in jieba.cut(string):
        if char=='':
            continue
        if len(char)==1:
            seg_feature.append(0)
        else:
            seg=[2]*len(char)
            seg[0]=1
            seg[-1]=3
            seg_feature.append(seg)
    return seg_feature

class BatchManager(object):
    def __init__(self,data,batch_size):
        self.batch_data=self.sort_and_pad(data,batch_size)
        self.len_data=len(self.batch_data)
    def sort_and_pad(self,data,batch_size):
        num_batch=int(math.cell(len(data)/batch_size))
        sorted_data=sorted(data,key=lambda x: len(x[0]))
        batch_data=list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size):(i+1)*int(batch_size)]))
        return batch_data
    @staticmethod
    def pad_data(data):
        strings=[]
        chars=[]
        segs=[]
        targets=[]
        max_length=max([len(sentence) for sentence in data])
        for line in data:
            string,char,seg,tag=line
            padding=[0]*(max_length-len(string))
            strings.append(string+padding)
            chars.append(char+padding)
            segs.append(seg+padding)
            targets.append(tag+padding)
        return [strings,chars,segs,targets]

    def iter_batch(self,shuffie=False):
        if shuffie:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]