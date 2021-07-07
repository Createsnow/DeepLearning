# -*- coding:utf-8 -*-

import json
import os
import re
import logging
import numpy as np
import tensorflow as tf

def save_config(file,file_dir):
    with open(file_dir,"w",encoding="utf-8") as fp:
        json.dump(fp,file,indent=4,ensure_ascii=False)

def load_config(file_dir):
    with open(file_dir,"r",encoding='utf-8') as fp:
        json.load(fp)

def make_check(params):
    if not os.path.exists(params.result_file):
        os.makedirs(params.result_file)
    if not os.path.exists(params.ckpt_file):
        os.makedirs(params.ckpt_file)
    if not os.path.exists("log"):
        os.makedirs("log")

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))

def load_vec(old_weigth,vet_path,emb_dim,id_to_char):
    new_weigth=old_weigth
    pre_train={}
    error_data=0
    with open(vet_path,"r",encoding="utf-8") as file:
        for line in file:
            word=line.strip().split(" ")
            if len(word)==emb_dim+1:
                pre_train[word[0]]=np.array([float(i) for i in word[1:]]).astype(np.float32)
            else:
                error_data+=1
    if error_data>0:
        print("WARNING:%d invalid line"%error_data)
    c_found=0
    c_lower=0
    c_zero=0
    for i in range(len(id_to_char)):
        if id_to_char[i] in pre_train:
            new_weigth[i]=pre_train[id_to_char[i]]
            c_found+=1
        elif id_to_char[i].lower() in pre_train:
            new_weigth[i]=pre_train[id_to_char[i]]
            c_lower+=1
        elif re.sub('\d','0',id_to_char[i].lower()) in pre_train:
            new_weigth[i]=pre_train[id_to_char[i]]
            c_zero+=1
    return new_weigth

def create_model(sess,model_class,config,model_path,id_to_char,logger):
    model=model_class(config)
    ckpt=tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model paramenter from %s" %(ckpt.model_checkpoint_path))
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logger.info("Create model flesh parametes!")
        sess.run(tf.global_variables_initializer())
    if config["pre_emb"]:
        #获取模型初始参数。
        old_emb_weigths=sess.run(model.char_lookup.read_value())
        #获取得到新的模型参数。
        new_emb_weigths=load_vec(old_emb_weigths,config["emb_path_file"],config["lstm_dim"],id_to_char)
        sess.run(model.char_lookup.assign(new_emb_weigths))
        logger.info("Load pre embedding!")
    return model

if __name__=="__main__":
    vec_path='./data/vec.txt'
    weigth=None
    id_to_char=None
    char_dim=None
    load_vec(weigth,vec_path,char_dim,id_to_char)