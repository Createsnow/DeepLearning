# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn


def clasvc():
    pass


class model(object):
    def __init__(self, config):

        self.config = config
        #批次大小
        self.batch_size = config["batch_size"]
        #字向量长度
        self.char_dim = config["char_dim"]
        #分词向量长度
        self.seg_dim = config["seg_dim"]
        #细胞隐藏层维度大小
        self.hidden_unit = config["lstm_dim"]
        #子的个数
        self.num_char = config["num_char"]
        #选用模型类型
        self.model_type = config["model_type"]
        #dropout
        self.dropout_rate = config["dropout_rate"]
        #细胞层数
        self.num_layer = config["num_layer"]

        #分词表示数量
        self.n_seg = 4
        #词嵌入向量长度
        self.word_dim=self.seg_inputs+self.char_dim
        #初始化器去初始化参数
        self.initializer = initializers.xavier_initializer()

        self.char_inputs = tf.placeholder(dype=tf.int32, shape=[None, None], name="char_inputs")
        self.seg_inputs = tf.placeholder(dype=tf.int32, shape=[None, None], name="seg_inputs")
        self.targets = tf.placeholder(dype=tf.int32, shape=[None, None], name="targets")

        #批次长度
        self.length = tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.char_inputs)), axis=-1), tf.float32)
        #词嵌入，ID转向量
        self.embedding = self.embedding_layer(self.char_inputs, self.seg_inputs)
        #模型选用
        if self.model_type == "bilstm":
            #网络模型构建，训练
            model_output = self.lstm_layer(self.embedding, self.hidden_unit,self.dropout_rate)
            #最后一层做全连接
            self.logit=self.project_lstm_layer(model_output)
        elif self.model_type == "idcnn":
            model_output = self.idcnn_layer()
            self.logit = self.project_lstm_layer(model_output)
        else:
            raise KeyError
        self.loss=self.loss_layer(self.logit,self.targets)
    def embedding_layer(self, char_input, seg_input):
        # 字向量信息【B,T,100】
        with tf.variable_scope("char_embedding"):
            self.char_embedding = tf.get_variable(name="char_embedding", shape=[self.num_char, self.char_dim],
                                                  dype=tf.float32,
                                                  initializer=self.initializer)
            char_emb = tf.nn.embedding_lookup(self.char_embedding, char_input, name="char_emb")

        # 词向量信息【B,T,20】
        with tf.variable_scope("seg_embedding"):
            self.seg_embedding = tf.get_variable(name="seg_input", shape=[self.n_seg, self.seg_dim], dtype=tf.float32,
                                                 initializer=self.initializer)
            seg_emb = tf.nn.embedding_lookup(self.seg_embedding, seg_input, name="seg_embedding")

        embedding = tf.concat([char_emb, seg_emb], axis=-1)
        return embedding

    def lstm_layer(self, model_input, hidden_unit, dropout_rate):
        cell_fw = rnn.BasicLSTMCell(hidden_unit)
        cell_bw = rnn.BasicLSTMCell(hidden_unit)
        with tf.variable_scope("lstm_layer"):
            if self.dropout_rate:
                cell_fw = rnn.DropoutWrapper(cell_fw, dropout_rate)
                cell_bw = rnn.DropoutWrapper(cell_bw, dropout_rate)
            if self.num_layer:
                cell_fw = rnn.MultiRNNCell([cell_fw for _ in self.num_layer],state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw for _ in self.num_layer],state_is_tuple=True)

            (output_fw,output_bw),_ = rnn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                    cell_bw=cell_bw,
                                                                    inputs=model_input,
                                                                    sequence_length=self.length,
                                                                    dtype=tf.float32)
            output=tf.concat([output_fw,output_bw],axis=-1)
            return output

    def idcnn_layer(self):
        pass

    def project_idcnn_layer(self):
        pass



    def project_lstm_layer(self,model_output):

        model_output=tf.expand_dims(model_output,1)
        with tf.variable_scope("project_lstm_layer"):
            w=tf.get_variable(name='w',shape=[self.batch_size,1,self.word_dim*2,self.word_dim],
                              dtype=tf.float32,
                              initializer=self.initializer)
            b=tf.get_variable(name='b',shape=[self.word_dim],dtype=tf.float32)
            logit=tf.nn.xw_plus_b(model_output,w,b)
        return logit

    def loss_layer(self,logit,label):
        pass

    def feet_batch(self):
        pass

    def run(self):
        pass

    pass
