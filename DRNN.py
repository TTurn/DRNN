# -*- coding: UTF-8 -*-
# python3.5(tensorflow)：C:\Users\Dr.Du\AppData\Local\conda\conda\envs\tensorflow\python.exe
# python3.6：C:\ProgramData\Anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 23:19
# @Author  : tuhailong

import tensorflow as tf

class DRNN():
    def __init__(self, sequence_max_length, embedding_size, num_classes, vocab_size, window_size, rnn_model, hidden_size, batch_size):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_max_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.rnn_model = rnn_model
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="embedding_W")
            self.embedded_input = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
            print("-"*20)
            print("Embedded Lookup:", self.embedded_input.get_shape())
            print("-"*20)

        pad_input = tf.pad(self.input_x, [[0, 0], [2, 0]])
        print("pad_input:", self.embedded_input.get_shape())

        state = None
        outputs = []
        for i in range(sequence_max_length):
            rnn_input = tf.slice(pad_input, [0, i, 0], [-1, window_size, -1])
            if rnn_model == 'RGU':
                output, state = self.GRU(rnn_input, self.hidden_size, state)
                outputs.append(output)

    def GRU(self, input_x, hidden_size, initial_state):
        gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        if initial_state == None:
            initial_state = gru_cell.zero_state(self, self.batch_size)
        outputs, state = tf.nn.dynamic_rnn(gru_cell, input_x, initial_state=initial_state)
        return outputs[-1], state

    def LSTM(self, input_x, hidden_size, initial_state):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        if initial_state == None:
            initial_state = lstm_cell.zero_state(self, self.batch_size)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, input_x, initial_state=initial_state)
        return outputs[-1], state