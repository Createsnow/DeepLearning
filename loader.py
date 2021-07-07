# -*- coding:utf-8 -*-


import os
import re
from data_utils import create_mapping
import codecs

root=os.getcwd()+os.sep
data_dir=os.path.join(root,'data/example.train')
def load_sentences(data_dir,lower,zero,*args,**kwargs):
    sentences=[]
    sentence=[]
    for line in codecs.open(data_dir,'r',encoding='utf-8'):
        if not line:
            if len(sentence)>0:
                if 'DOCSTARA' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence=[]
        else:
            if line[0]==' ':
                line="$"+line[1:0]
                word=line.split()
            else:
                word=line.split("\t")
            assert len(word)==2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTARA' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def update_tag_schema(sentences,tag_schema):
    for i,s in enumerate(sentences):
        tags=[w[-1] for w in s]
        if not iob2(tags):
            s_str='\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format'+'please check senteces %i:\n%s'%(i,s_str))

        if tag_schema=='iob':
            for word,new_tag in zip(s,tags):
                word[-1]=new_tag
        elif tag_schema=='iobes':
            new_tags=iob_to_iobes(tags)
            for word, new_tag in zip(s,new_tags):
                word[-1]=new_tag
        else:
            raise  Exception('Unknown tagging schemae!')

def iob_to_iobes(tags):
    new_tags=[]
    for i,tag in enumerate(tags):
        if tag=='O':
            new_tags.append(tag)
        elif tag.split('-')[0]=='B':
            if i+1<len(tags) and tags[i+1].split('-')[0]=='I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-','S-'))
        elif tag.split('-')[0]=='I':
            if i+1<len(tags) and tags[i+1].split('-')[0]=='I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-','E-'))
        else:
            raise Exception('Invalid IOB foramt!')
    return new_tags

def iob2(tags):
    for i,tag in enumerate(tags):
        if tag=='O':
            continue
        split=tag.split('-')
        if len(split) !=2 or split[0] not in ['B','I']:
            return False
        if split[0]=='B':
            continue
        elif i==0 or tags[i-1]=='O':
            tags[i]='B'+tag[1:]
        elif tags[i-1][1:]==tag[1:]:
            continue
        else:
            tags[i]='B'+tag[1:]
    return True
def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    #print('Loading pretrained embeddings from %s...' % ext_emb_path)
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


if __name__ == '__main__':
    load_sentences(data_dir,zero=None,lower=None)
    pass