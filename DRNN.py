# -*- coding: UTF-8 -*-
# python3.5(tensorflow)：C:\Users\Dr.Du\AppData\Local\conda\conda\envs\tensorflow\python.exe
# python3.6：C:\ProgramData\Anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 23:19
# @Author  : tuhailong

import tensorflow as tf
linear_initial = tf.contrib.layers.xavier_initializer()
class DRNN():

    def BN(selfself, inputs, is_training):
        return inputs

    def MLP(self, inputs, weight, bias):
        inputs = tf.add(tf.matmul(inputs, weight), bias)
        inputs = tf.nn.relu(inputs)
        return inputs

    def Maxpooling(self, inputs):
        pool_result = tf.nn.top_k(inputs, 1)[0]
        return pool_result

    def __init__(self, sequence_max_length, embedding_size, num_classes, vocan_size, window_size, rnn_model, hidden_size,
                 batch_size, mlp_size, keep_props, l2_reg_lambda):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None,num_classes], name="input_y")
        self.is_trianing = tf.placeholder(tf.bool)

        self.rnn_model = rnn_model
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.mlp_size = mlp_size
        self.keep_props = keep_props
        self.l2_reg_lambda = l2_reg_lambda
        self.num_classes = num_classes

        l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'), tf.name_scope("embedding layer"):
            self.embedding_W = tf.Variable(tf.random_uniform([vocan_size, embedding_size], -1.0, 1.0), name="embedding_W")
            self.embedding_input = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
        print("embedding lookup:", self.embedding_input.get_shape())

        pad_input = tf.pad(self.embedding_input, [[0, 0], [window_size-1, 0], [0, 0]], mode="CONSTANT")
        print("pad_input:", pad_input.get_shape())

        rnn_inputs = []
        for i in range(sequence_max_length):
            rnn_inputs.append(tf.slice(pad_input, [0, i, 0], [-1, window_size, -1], name='rnn_input'))
        rnn_input_tensor = tf.concat(rnn_inputs, 1)
        print("rnn_input_tensor:", rnn_input_tensor.get_shape())

        if self.rnn_model == 'GRU':
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        elif self.rnn_model == 'LSTM':
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        else:
            raise ValueError("invalid rnn model")

        with tf.variable_scope("drnn"):
            if self.is_trianing == True:
                self.rnn_cell = tf.nn.rnn_cell.DropoutWrapper(self.rnn_cell, input_keep_prob=keep_props[0], output_keep_prob=keep_props[1], state_keep_prob=keep_props[2])
            outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, rnn_input_tensor, dtype=tf.float32)
            output_stack = tf.stack(outputs, axis=0)
            print("output_stack:", output_stack.get_shape())
            output_trans = tf.transpose(output_stack, [1, 0, 2])
            print("output_trans:", output_trans.get_shape())
            output_unstack = tf.unstack(output_trans, axis=0)

            outputs_list = []
            weight = tf.get_variable('w', [self.hidden_size, self.mlp_size], initializer=linear_initial)
            bias = tf.get_variable('b', [self.mlp_size], initializer=tf.constant_initializer(0.0))
            for i in range(len(output_unstack)):
                if (i+1) % 3 == 0:
                    #output = self.BN(output_unstack, self.is_training)
                    if i == 2:
                        print("output_unstack[i]:", output_unstack[i].get_shape())
                    output = self.MLP(output_unstack[i], weight, bias)
                    if i == 2:
                        print("MLP:", output.get_shape())
                    output = self.Maxpooling(output)
                    if i == 2:
                        print("maxpool:", output.get_shape())
                    outputs_list.append(output)
            l2_loss += tf.nn.l2_loss(weight) + tf.nn.l2_loss(bias)
            output = tf.concat(outputs_list, 1)
            print("drnn:", output.get_shape())

            #calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                self.pretictions = tf.argmax(output, 1, name="predictions")
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.input_y)
                self.l2_loss = self.l2_reg_lambda * l2_loss
                self.loss = tf.reduce_mean(losses) + self.l2_loss

            #accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.pretictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name="accuracy")

            print("model init finish")



























