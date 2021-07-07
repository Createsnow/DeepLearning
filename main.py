# -*- coding:utf-8 -*-

import os,pickle,logging
import itertools
from collections import OrderedDict
from utils import load_config,make_check,save_config,get_logger,create_model,print_config
from loader import load_sentences,update_tag_schema,augment_with_pretrained
from data_utils import char_mapping,tag_mapping,prepare_dataset,BatchManager
import tensorflow as tf

root = os.getcwd() + os.sep

tf.flags.DEFINE_boolean("clean", False, "clean train folder")
tf.flags.DEFINE_boolean("train", True, "whether train the model!")
tf.flags.DEFINE_integer("seg_dim", 28, "Embedding size of segmentation ,0 if not use")
tf.flags.DEFINE_integer("char_dim", 100, "Embedding size of characters")
tf.flags.DEFINE_integer("lstm_dim", 128, "num of hidden units in LSTM or filters in IDCNN")
tf.flags.DEFINE_string("tag_schema", 'iobes', "tag schema use iob or iobes ")

tf.flags.DEFINE_integer("num_layer", 2, "cell layers")
tf.flags.DEFINE_float("clip", 5, "Gradient clip")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout rate")
tf.flags.DEFINE_float("batch_size", 32, "batch size")
tf.flags.DEFINE_float("learning", 0.0001, "initial learning rate")
tf.flags.DEFINE_string("optimizer", 'adam', "optimizer for training")
tf.flags.DEFINE_boolean("pre_emb", True, "wither use pre_trained embedding ")
tf.flags.DEFINE_boolean("zeros", True, "wither replace digits with zero")
tf.flags.DEFINE_boolean("lower", False, "wither lower case")

tf.flags.DEFINE_integer("max_epoch", 100, "maximum training epoch")
tf.flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")
tf.flags.DEFINE_string("check_file", "ckpt", "path to save model")
tf.flags.DEFINE_string("summary_file", "summary", "path to store summaries")
tf.flags.DEFINE_string("log_file", "train.log", "log file of train")
tf.flags.DEFINE_string("map_file", "maps.pkl", "file for maps")
tf.flags.DEFINE_string("vocab_file", "vecab.json", "file for vocab")
tf.flags.DEFINE_string("config_file", "config.json", "file for config")
tf.flags.DEFINE_string("script", "conlleval", "evaluation script")
tf.flags.DEFINE_string("result_file", "result", "path for model result")
tf.flags.DEFINE_string("emb_path_file", os.path.join(root, "data/vec.txt"), "path for pre_trained embeding")
tf.flags.DEFINE_string("train_path_flie", os.path.join(root, "data/example.train"), "path for train data")
tf.flags.DEFINE_string("test_path_file", os.path.join(root, "data/example.test"), "path for test data")
tf.flags.DEFINE_string("dev_path_flie", os.path.join(root, "data/example.dev"), "path for dev data")
tf.flags.DEFINE_string("model_type", "lstm", "model of lstm or IDCNN")

FLAGS = tf.flags.FLAGS
# 验证
assert FLAGS.clip < 5.1, "gradient clip should’t be too much"
assert 0 < FLAGS.droport < 1, "dropout rate between 0 and 1"
assert FLAGS.learning > 0, "learning rate must larger than zero"
assert FLAGS.optimzer in ["adam", "sgd", "adagrad"]


# confilg for the model
def model_config(char_to_id, tag_to_id):
    config = OrderedDict()
    config["seg_dim"] = FLAGS.seg_dim
    config["char_dim"] = FLAGS.char_dim
    config["lstm_lstm"] = FLAGS.lstm_dim
    config["num_char"] = len(char_to_id)
    config["num_tag"] = len(tag_to_id)
    config["model_type"] = FLAGS.model_type
    config["batch_size"] = FLAGS.batch_size
    config["emb_file"] = FLAGS.emb_path_file
    config["clip"] = FLAGS.clip
    config["learning_rate"] = FLAGS.learning
    config["dropout_rate"] = FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["tag_schema"] = FLAGS.schema
    config["pre_emb"] = FLAGS.pre_emb
    config["lower"] = FLAGS.lower
    config["zero"] = FLAGS.zero
    config["num_layer"]=FLAGS.num_layer
    return config


def train():
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zero)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zero)
    dev_sentences = load_sentences(FLAGS.dev_dile, FLAGS.lower, FLAGS.zero)
    #转为IBOES格式的标签。
    update_tag_schema(train_sentences,FLAGS.tag_schema)

    if not os.path.isfile(FLAGS.map_file):
        if FLAGS.pre_emb:
            dico_chars_train=char_mapping(train_sentences,FLAGS.lower)[0]
            dico_chars,char_to_id,id_to_char=augment_with_pretrained(
                                    dico_chars_train.copy(),
                                    FLAGS.emb_file,
                                    list(itertools.chain.from_iterable(
                                        [[ s[0] for s in sentence]for sentence in test_sentences]
                                    )

                                    )
            )
        else:
             _,char_to_id,id_to_char=char_mapping(train_sentences,FLAGS.lower)

        _,tag_to_id,id_to_tag=tag_mapping(train_sentences)

        with open(root+FLAGS.maps_file,'wb',encoding='utf-8') as mp:
            pickle.dump([char_to_id,id_to_char,tag_to_id,id_to_tag],mp)
    else:
        with open(root+FLAGS.map_file,'rb',encoding='utf-8') as f:
            char_to_id,id_to_char,tag_to_id,id_to_tag=pickle.load(f)

    #数据预处理
    train_data=prepare_dataset(train_sentences,char_to_id,tag_to_id,FLAGS.lower)
    test_data=prepare_dataset(test_sentences,char_to_id,tag_to_id,FLAGS.lower)
    dev_data=prepare_dataset(dev_sentences,char_to_id,tag_to_id,FLAGS.lower)
    logging.info('train/test/dev data:%d/%d/%d'%(len(train_data),len(test_data),len(dev_data)))

    train_manager=BatchManager(train_data,FLAGS.batch_size)
    test_manager=BatchManager(test_data,FLAGS.batch_size)
    dev_manager=BatchManager(dev_data,FLAGS.batch_size)

    make_check(FLAGS)
    if not os.path.isfile(FLAGS.config_file):
        config=model_config(char_to_id,tag_to_id)
        save_config(config,FLAGS.config_file)
    else:
        config=load_config(FLAGS.config_file)

    log_path_file=os.path.join("log",FLAGS.log_file)
    logger=get_logger(log_path_file)
    print_config(config, logger)

    tf_config=tf.ConfigProto(log_device_placment=True)
    tf_config.gpu_options.allow_growth=True
    with tf.Session(config=tf_config) as sess:
        model=create_model(sess,model,config,FLAGS.model_file,char_to_id,logger)

