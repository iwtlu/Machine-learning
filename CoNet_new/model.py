import numpy as np
import tensorflow as tf
from past.builtins import xrange
from Dataset import Dataset
from collections import defaultdict
import time
import sys
import math
import random
import os



class MTL(object):
    def __init__(self, config, sess):
        t1 = time.time()
        # data corpus and load data: multi-task
        self.data_dir = config['data_dir']

        # 1) task: app rec
        self.data_name_app = config['data_name_app']
        dataset_app = Dataset(self.data_dir + self.data_name_app)
        self.train_app, self.testRatings_app = dataset_app.trainMatrix, dataset_app.testRatings     # 用户*物品
        self.nUsers_app, self.nItems_app = dataset_app.nUsers, dataset_app.nItems
        self.U_app, self.V_app = dataset_app.U, dataset_app.V
        # self.nUsers_app_train, self.nItems_app_train = self.train_app.shape
        # self.nUsers_app_test, self.nItems_app_test = self.testRatings_app.shape
        # self.nUsers_app, self.nItems_app = max(self.nUsers_app_train, self.nUsers_app_test), max(self.nItems_app_train,
        #                                                                                          self.nItems_app_test)
        print("----Load app data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          %(time.time()-t1, self.nUsers_app, self.nItems_app, self.train_app.nnz, self.testRatings_app.nnz))
        self.user_input_app, self.item_input_app, self.labels_app = [], [], []
        self.test_user_input_app, self.test_item_input_app, self.test_labels_app = [], [], []

        # 2) task: news rec
        self.data_name_news = config['data_name_news']
        dataset_news = Dataset(self.data_dir + self.data_name_news)
        self.train_news, self.testRatings_news = dataset_news.trainMatrix, dataset_news.testRatings
        self.nUsers_news, self.nItems_news = dataset_news.nUsers, dataset_news.nItems
        self.U_news, self.V_news = dataset_news.U, dataset_news.V
        # self.nUsers_news_train, self.nItems_news_train = self.train_news.shape
        # self.nUsers_news_test, self.nItems_news_test = self.testRatings_news.shape
        # self.nUsers_news, self.nItems_news = max(self.nUsers_news_train, self.nUsers_news_test), max(self.nItems_news_train, self.nItems_news_test)

        print("----Load news data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          %(time.time()-t1, self.nUsers_news, self.nItems_news, self.train_news.nnz, self.testRatings_news.nnz))
        self.user_input_news, self.item_input_news, self.labels_news = [],[],[]
        self.test_user_input_news, self.test_item_input_news, self.test_labels_news = [],[],[]

        if self.nUsers_app != self.nUsers_news:     # 要求总用户数必须相同
            print('nUsers_app != nUsers_news. However, they should be shared. exit...')
            sys.exit(0)
        self.nUsers = self.nUsers_news

        # hyper-parameters
        self.init_std = config['init_std']
        self.batch_size = config['batch_size']
        self.nepoch = config['nepoch']
        self.layers = config['layers']
        self.edim_u = config['edim_u']
        self.edim_v = config['edim_v']
        self.edim = self.edim_u + self.edim_v  # concat
        self.nhop = len(self.layers)
        self.max_grad_norm = config['max_grad_norm']
        self.negRatio = config['negRatio']
        self.activation = config['activation']
        self.learner = config['learner']
        self.objective = config['objective']
        self.class_size = 1  # 真实评分
        self.input_size = 2  # (user, item) 输入的是用户 物品，竟然没有评分？？？
        # save and restore
        self.show = config['show']
        self.checkpoint_dir = config['checkpoint_dir']

        # 神经网络输入层、输出层---空层
        self.input_app = tf.placeholder(tf.int32, [self.batch_size, self.input_size], name="input")  # batch_size * (user, app)
        self.target_app = tf.placeholder(tf.float32, [self.batch_size, self.class_size], name="target")
        self.input_news = tf.placeholder(tf.int32, [self.batch_size, self.input_size], name="input")  # (user, news)
        self.target_news = tf.placeholder(tf.float32, [self.batch_size, self.class_size], name="target")

        self.lr = None
        self.init_lr = config['init_lr']
        self.current_lr = config['init_lr']
        self.loss_joint = None
        self.loss_app_joint = None
        self.loss_news_joint = None
        self.loss_app_only = None
        self.loss_news_only = None
        self.optim_joint = None
        self.optim_app = None
        self.optim_news = None
        self.step = None
        self.sess = sess
        self.log_loss_app = []
        self.log_perp_app = []
        self.log_loss_news = []
        self.log_perp_news = []
        self.isDebug = config['isDebug']
        self.isOneBatch = config['isOneBatch']

        # multi-task
        self.weights_app_news = config['weights_app_news']
        self.cross_layers = config['cross_layers']
        assert self.cross_layers > 0 and self.cross_layers < self.nhop

        # evaluation
        self.bestrmse_app = 10086
        self.bestrmse_epoch_app = -1
        self.bestrmse_news = 10086
        self.bestrmse_epoch_news = -1
        self.rmse_app, self.rmse_news = 0, 0

    # 共享参数
    def build_memory_shared(self):
        ## ------- parameters: shared-------  ##
        # 1. embedding matrices for input <user, app>, <user, news>: shared user embedding matrix 输入的嵌入矩阵
        # self.U = tf.Variable(tf.random_normal([self.nUsers, self.edim_u], stddev=self.init_std))  # 用户特征矩阵 sharing user factors 服从正太分布
        # 2. match the dimensions... 维度匹配
        self.shared_Hs = defaultdict(object)    # 设置对象 包含多层
        for h in xrange(1, self.cross_layers+1):  # only cross between h=1 2 64*32---32*16
            self.shared_Hs[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))

    # 一个域的私有参数
    def build_memory_app_specific(self):
        ## ------- parameters: app specific -------  ##
        # 1. embedding matrices for input <user, app>, <user, news>: shared user embedding matrix
        # self.V_app = tf.Variable(tf.random_normal([self.nItems_app, self.edim_v], stddev=self.init_std))    # 物品嵌入矩阵 物品*维度
        # 2. weights & biases for hidden layers: the input to hidden layers are the merged embedding app域的权重、偏置
        self.weights_app = defaultdict(object)
        self.biases_app = defaultdict(object)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            self.weights_app[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))
            self.biases_app[h] = tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std))
        # 3. output layer: weight and bias
        self.h_app = tf.Variable(tf.random_normal([self.layers[-1], self.class_size], stddev=self.init_std))
        self.b_app = tf.Variable(tf.random_normal([self.class_size], stddev=self.init_std))

    # 单独训练一个域：app
    def build_model_app_training(self):
        ## ------- computational graph: app training only ------- ##
        # 1. input & embedding layer 输入层
        USERin_app = tf.nn.embedding_lookup(self.U_app, self.input_app[:, 0])    # input_app:128*(u,i) 获取输入的用户特征  U:128*32
            # 3D due to batch  tf.nn.embedding_lookup 张量查找 在U中寻找input_app用户对应的行
        ITEMin_app = tf.nn.embedding_lookup(self.V_app, self.input_app[:, 1])    # 对应块的物品*32  128*32
        UIin_app = tf.concat(values=[USERin_app, ITEMin_app], axis=1)  # no info loss, and edim = edim_u + edim_v 128*64
        # print(' app concat', UIin_app.shape)

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
        self.layer_h_apps = defaultdict(object)
        layer_h_app = tf.reshape(UIin_app, [-1, self.edim])  # init: merged embedding 重塑成二维， 行未知*列64 代表l+1层的结果
        # print(' app concat', UIin_app.shape, 'resape', layer_h_app.shape)
        self.layer_h_apps[0] = tf.to_float(layer_h_app)  # tf.identity(layer_h_app) 代表l层的结果
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers 执行隐层操作
            layer_h_app = tf.add(tf.matmul(self.layer_h_apps[h-1], self.weights_app[h]), self.biases_app[h])
            if self.activation == 'relu':
                layer_h_app = tf.nn.relu(layer_h_app)
            elif self.activation == 'sigmoid':
                layer_h_app = tf.nn.sigmoid(layer_h_app)

            self.layer_h_apps[h] = layer_h_app  #tf.identity(layer_h_app)
            # layer_h = tf.nn.dropout(layer_h, keep_prob) https://www.tensorflow.org/get_started/mnist/pros
        # 'layer_h' is now the representations of last hidden layer

        # 3. output layer: dense and linear 输出层
        self.z_app_only = tf.matmul(layer_h_app, self.h_app) + self.b_app
        # self.pred_app_only = tf.nn.softmax(self.z_app_only)     # ！！！！！！！！！！！要改
        self.pred_app_only = tf.identity(self.z_app_only)

        ## ------- loss and optimization ------- ##
        if self.objective == 'cross':
            self.loss_app_only = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_app_only, labels=self.target_app)
            #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.target)
        elif self.objective == 'log':
            self.loss_app_only = tf.losses.log_loss(predictions=self.pred_app_only, labels=self.target_app)
        elif self.objective == 'mse':   # mse损失函数
            self.loss_app_only = tf.losses.mean_squared_error(predictions=self.pred_app_only, labels=self.target_app)
        else:
            self.loss_app_only = tf.losses.hinge_loss(logits=self.z_app_only, labels=self.target_app)

        self.lr = tf.Variable(self.current_lr)  # 学习率
        if self.learner == 'adam':  # ！！！！这个即可
            self.opt_app = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_app = tf.train.RMSPropOptimizer(learning_rate=self.lr,decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_app = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_app = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        # params = [self.U, self.V_app, self.h_app, self.b_app]   # 获取参数
        params = [self.h_app, self.b_app]  # 获取参数
        for h in range(1, self.nhop):  # weighs/biases in hidden layers
            params.append(self.weights_app[h])
            params.append(self.biases_app[h])
        grads_and_vars = self.opt_app.compute_gradients(self.loss_app_only, params)     # 利用损失函数优化参数
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]     # 梯度规约 防止梯度爆炸

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            #self.optim = self.opt.apply_gradients(grads_and_vars)
            self.optim_app = self.opt_app.apply_gradients(clipped_grads_and_vars)

    def build_memory_news_specific(self):
        ## ------- parameters: news specific -------  ##
        # 1. embedding matrices for input <user, app>, <user, news>: shared user embedding matrix
        # self.V_news = tf.Variable(tf.random_normal([self.nItems_news, self.edim_v], stddev=self.init_std))

        # 2. weights & biases for hidden layers: the input to hidden layers are the merged embedding
        self.weights_news = defaultdict(object)
        self.biases_news = defaultdict(object)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            self.weights_news[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))
            self.biases_news[h] = tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std))
        # 3. output layer: weight and bias
        self.h_news = tf.Variable(tf.random_normal([self.layers[-1], self.class_size], stddev=self.init_std))
        self.b_news = tf.Variable(tf.random_normal([self.class_size], stddev=self.init_std))

    def build_model_news_training(self):
        ## ------- computational graph: news training only ------- ##
        # 1. input & embedding layer
        USERin_news = tf.nn.embedding_lookup(self.U_news, self.input_news[:, 0])
        ITEMin_news = tf.nn.embedding_lookup(self.V_news, self.input_news[:, 1])
        UIin_news = tf.concat(values=[USERin_news, ITEMin_news], axis=1)  # no info loss, and edim = edim_u + edim_v

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
        self.layer_h_newss = defaultdict(object)
        layer_h_news = tf.reshape(UIin_news, [-1, self.edim])  # init: cmerged embedding
        self.layer_h_newss[0] = tf.cast(layer_h_news, dtype=tf.float32)  # tf.identity(layer_h_news)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            print(type(self.layer_h_newss[h-1]), type(self.weights_news[h]))
            layer_h_news = tf.add(tf.matmul(self.layer_h_newss[h-1], self.weights_news[h]), self.biases_news[h])

            if self.activation == 'relu':
                layer_h_news = tf.nn.relu(layer_h_news)
            elif self.activation == 'sigmoid':
                layer_h_news = tf.nn.sigmoid(layer_h_news)
            self.layer_h_newss[h] = layer_h_news
            # layer_h = tf.nn.dropout(layer_h, keep_prob) https://www.tensorflow.org/get_started/mnist/pros
        # 'layer_h' is now the representations of last hidden layer

        # 3. output layer: dense and linear
        self.z_news_only = tf.matmul(layer_h_news, self.h_news) + self.b_news
        #  self.pred_news_only = tf.nn.softmax(self.z_news_only)
        self.pred_news_only = tf.identity(self.z_news_only)

        ## ------- loss and optimization ------- ##
        if self.objective == 'cross':
            self.loss_news_only = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_news_only, labels=self.target_news)
            #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.target)
        elif self.objective == 'log':
            self.loss_news_only = tf.losses.log_loss(predictions=self.pred_news_only, labels=self.target_news)
        elif self.objective == 'mse':   # mse损失函数
            self.loss_news_only = tf.losses.mean_squared_error(predictions=self.pred_news_only, labels=self.target_news)
        else:
            self.loss_news_only = tf.losses.hinge_loss(logits=self.z_news_only, labels=self.target_news)

        self.lr = tf.Variable(self.current_lr)
        if self.learner == 'adam':
            self.opt_news = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_news = tf.train.RMSPropOptimizer(learning_rate=self.lr,decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_news = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_news = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        # params = [self.U, self.V_news, self.h_news, self.b_news]
        params = [self.h_news, self.b_news]
        for h in range(1, self.nhop):  # weighs/biases in hidden layers
            params.append(self.weights_news[h])
            params.append(self.biases_news[h])
        grads_and_vars = self.opt_news.compute_gradients(self.loss_news_only, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            #self.optim = self.opt.apply_gradients(grads_and_vars)
            self.optim_news = self.opt_news.apply_gradients(clipped_grads_and_vars)

    # 训练公共块
    def build_model_joint_training(self):
        ## ------- computational graph: for joint training ------- ##
        # 1. input & embedding layer
        USERin_app = tf.nn.embedding_lookup(self.U_app, self.input_app[:,0])  # 3D due to batch self.input_app[:,0]作为索引返回U中的张量
        USERin_news = tf.nn.embedding_lookup(self.U_news, self.input_news[:,0])

        ITEMin_app = tf.nn.embedding_lookup(self.V_app, self.input_app[:,1])
        ITEMin_news = tf.nn.embedding_lookup(self.V_news, self.input_news[:,1])
        UIin_app = tf.concat(values=[USERin_app, ITEMin_app], axis=1)  # no info loss, and edim = edim_u + edim_v
        UIin_news = tf.concat(values=[USERin_news, ITEMin_news], axis=1)  # no info loss, and edim = edim_u + edim_v

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
        # cross computation between app network and news network 跨域
        self.layer_h_apps = defaultdict(object)
        layer_h_app = tf.reshape(UIin_app, [-1, self.edim])  # init: merged embedding
        self.layer_h_apps[0] = tf.to_float(layer_h_app)

        self.layer_h_newss = defaultdict(object)
        layer_h_news = tf.reshape(UIin_news, [-1, self.edim])  # init: merged embedding
        self.layer_h_newss[0] = tf.to_float(layer_h_news)

        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            # MLP层
            # 1) app-specific: o_app^t+1 = (W_app^t a_app^t + b_app^t) + a_news^t
            layer_h_app = tf.add(tf.matmul(self.layer_h_apps[h-1], self.weights_app[h]), self.biases_app[h])    # 128*32 本域
            if h <= self.cross_layers:  # 如果是在跨域层
                layer_h_app = tf.add(layer_h_app, tf.matmul(self.layer_h_newss[h-1], self.shared_Hs[h]))    # 跨域+本域

            if self.activation == 'relu':
                layer_h_app = tf.nn.relu(layer_h_app)
            elif self.activation == 'sigmoid':
                layer_h_app = tf.nn.sigmoid(layer_h_app)
            self.layer_h_apps[h] = layer_h_app  # l层经过激活函数后的结果

            # 2) news-specific:  o_news^t+1 = (W_news^t a_news^t + b_news^t) + a_app^t
            layer_h_news = tf.add(tf.matmul(self.layer_h_newss[h-1], self.weights_news[h]), self.biases_news[h])
            if h <= self.cross_layers:
                layer_h_news = tf.add(layer_h_news, tf.matmul(self.layer_h_apps[h-1], self.shared_Hs[h]))
            if self.activation == 'relu':
                layer_h_news = tf.nn.relu(layer_h_news)
            elif self.activation == 'sigmoid':
                layer_h_news = tf.nn.sigmoid(layer_h_news)
            self.layer_h_newss[h] = layer_h_news

        # 3. output layer: dense and linear
        self.z_app_joint = tf.matmul(layer_h_app, self.h_app) + self.b_app
        # self.pred_app_joint = tf.nn.softmax(self.z_app_joint)
        self.pred_app_joint = tf.identity(self.z_app_joint)
        self.z_news_joint = tf.matmul(layer_h_news, self.h_news) + self.b_news
        # self.pred_news_joint = tf.nn.softmax(self.z_news_joint)
        self.pred_news_joint = tf.identity(self.z_news_joint)

        ## ------- loss and optimization ------- ##
        # 1、app的损失函数
        if self.objective == 'cross':
            self.loss_app_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_app_joint, labels=self.target_app)
        elif self.objective == 'log':
            self.loss_app_joint = tf.losses.log_loss(predictions=self.pred_app_joint, labels=self.target_app)
        elif self.objective == 'mse':   # mse损失函数
            self.loss_app_joint = tf.losses.mean_squared_error(predictions=self.pred_app_joint, labels=self.target_app)
        else:
            self.loss_app_joint = tf.losses.hinge_loss(logits=self.z_app_joint, labels=self.target_app)
        # 2、news的损失函数
        if self.objective == 'cross':
            self.loss_news_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_news_joint, labels=self.target_news)
            #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.target)
        elif self.objective == 'log':
            self.loss_news_joint = tf.losses.log_loss(predictions=self.pred_news_joint, labels=self.target_news)
        elif self.objective == 'mse':   # mse损失函数
            self.loss_news_joint = tf.losses.mean_squared_error(predictions=self.pred_news_joint, labels=self.target_news)
        else:
            self.loss_news_joint = tf.losses.hinge_loss(logits=self.z_news_joint, labels=self.target_news)

        self.lr = tf.Variable(self.current_lr)
        if self.learner == 'adam':
            self.opt_joint = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_joint = tf.train.RMSPropOptimizer(learning_rate=self.lr,decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_joint = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_joint = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        # params = [self.U, self.V_app, self.h_app, self.b_app, self.V_news, self.h_news, self.b_news]
        params = [self.h_app, self.b_app, self.h_news, self.b_news]
        for h in range(1, self.nhop):  # weighs/biases in hidden layers
            params.append(self.weights_app[h])
            params.append(self.biases_app[h])
            params.append(self.weights_news[h])
            params.append(self.biases_news[h])
        for h in xrange(1, self.cross_layers+1):
            params.append(self.shared_Hs[h])  # only cross these layers
        # 最终的损失函数 = app + news
        self.loss_joint = self.weights_app_news[0] * self.loss_app_joint + self.weights_app_news[1] * self.loss_news_joint
        grads_and_vars = self.opt_joint.compute_gradients(self.loss_joint, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            #self.optim = self.opt.apply_gradients(grads_and_vars)
            self.optim_joint = self.opt_joint.apply_gradients(clipped_grads_and_vars)

    # 搭建神经网络模型
    def build_model(self):
        print('开始搭建神经网络模型')
        self.global_step = tf.Variable(0, name="global_step")

        self.build_memory_shared()  # 共享参数

        self.build_memory_app_specific()    # 域1特定参数 只有参数
        self.build_model_app_training()     # 设置数据传输方式

        self.build_memory_news_specific()   # 域2特定参数
        self.build_model_news_training()

        self.build_model_joint_training()
        tf.global_variables_initializer().run()

    def get_train_instances_app(self):
        # labeles:真实评分
        self.user_input_app, self.item_input_app, self.labels_app = [], [], []
        for (u, i) in self.train_app.keys():
            r = self.train_app[u, i]
            self.user_input_app.append(u)
            self.item_input_app.append(i)
            self.labels_app.append(r)

    def get_train_instances_news(self):
        self.user_input_news, self.item_input_news, self.labels_news = [],[],[]
        for (u, i) in self.train_news.keys():
            r = self.train_news[u, i]
            self.user_input_news.append(u)
            self.item_input_news.append(i)
            self.labels_news.append(r)

    # 得到测试集 需要修改 将矩阵存储进去即可
    def get_test_instances_app(self):
        self.test_user_input_app, self.test_item_input_app, self.test_labels_app = [],[],[]
        for (u, i) in self.testRatings_app.keys():
            r = self.testRatings_app[u, i]
            self.test_user_input_app.append(u)
            self.test_item_input_app.append(i)
            self.test_labels_app.append(r)

    def get_test_instances_news(self):
        self.test_user_input_news, self.test_item_input_news, self.test_labels_news = [],[],[]
        for (u, i) in self.testRatings_news.keys():
            r = self.testRatings_news[u, i]
            self.test_user_input_news.append(u)
            self.test_item_input_news.append(i)
            self.test_labels_news.append(r)

    def train_model(self):
        # 获取两个域的训练集： 用户集 物品集 评分集
        self.get_train_instances_app()  # randomly sample negatives each time / per epoch u i labels
        self.get_train_instances_news()  # randomly sample negatives each time / per epoch

        num_examples_app = len(self.labels_app)     # 总评分数
        num_batches_app = int(math.ceil(num_examples_app / self.batch_size))

        num_examples_news = len(self.labels_news)
        num_batches_news = int(math.ceil(num_examples_news / self.batch_size))

        num_batches_max = max(num_batches_app, num_batches_news)     # 最大块 最小块
        num_batches_min = min(num_batches_app, num_batches_news)
        # print('总块数: app={}, news={}, max={}, min={}'.format(num_batches_app, num_batches_news, num_batches_max, num_batches_min))

        x_app = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)  # (user,item)   输入：多个用户、物品对
        target_app = np.zeros([self.batch_size, self.class_size])  # 真实，每一个用户物品对 对应的评分

        x_news = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)     # 空数组
        target_news = np.zeros([self.batch_size, self.class_size])

        sample_ids_app = [sid for sid in range(num_examples_app)]   # 遍历训练集1。。。n
        random.shuffle(sample_ids_app)      # 打乱训练集顺序n..5..1：乱序的训练集编号
        sample_ids_news = [sid for sid in range(num_examples_news)]
        random.shuffle(sample_ids_news)

        cost_total = 0.0
        cost_joint = 0.0
        cost_app = 0.0
        cost_news = 0.0
        batches_app = [b for b in range(num_batches_app)]   # 1，2，，，，b块
        batches_news = [b for b in range(num_batches_news)]

        ## should first single training and then joint train (i.e. first pre-train and then fine-tune)
        # 利用多出来的块数进行预训练
        if num_batches_app < num_batches_news:  # app smaller than news, and hence single training on news first 先训练总块数大的那个
            # news块数多
            batches_join = batches_app
            batches_single = batches_news[num_batches_app:]     # news5个块 apps3个块，选择news的3个块
            # 先训练news多余块
            for _ in batches_single:    # 遍历多余块的每一个块
                target_news.fill(0)
                for b in xrange(self.batch_size):   # 遍历一个块内的所有用户、物品对 ：b = 1 2...128
                    if not sample_ids_news:  # if news used up 如果sample_ids_news空 最后剩下2个，但是块内取到了3 4 5.。。
                        sample_id_news = random.randrange(0, num_examples_news)  # 从所有的对中随机取一个
                        x_news[b][0] = self.user_input_news[sample_id_news]     # 往空数组里填充数据
                        x_news[b][1] = self.item_input_news[sample_id_news]
                        target_news[b] = self.labels_news[sample_id_news]
                    else:
                        sample_id_news = sample_ids_news.pop()  # 序号列表从列表头部开始出值，使用的id永远出栈，未使用的仍留在栈内
                        x_news[b][0] = self.user_input_news[sample_id_news]
                        x_news[b][1] = self.item_input_news[sample_id_news]
                        target_news[b] = self.labels_news[sample_id_news]

                keys = [self.input_news, self.target_news]  # 神经网络里的输入 真实层
                values = [x_news, target_news]
                _, loss_news, pred_news, self.step = self.sess.run([self.optim_news,
                                                    self.loss_news_only,
                                                    self.pred_news_only,
                                                    self.global_step],
                                                    feed_dict={
                                                        k:v for k,v in zip(keys, values)
                                                    })
                cost_news += np.sum(loss_news)
                cost_total += np.sum(loss_news)
        else:  # app is not smaller than news, and hence single training on app first
            batches_join = batches_news
            batches_single = batches_app[num_batches_news:]
            for _ in batches_single:
                target_app.fill(0)
                for b in xrange(self.batch_size):
                    if not sample_ids_app:  # if app used up
                        sample_id_app = random.randrange(0, num_examples_app)
                        x_app[b][0] = self.user_input_app[sample_id_app]
                        x_app[b][1] = self.item_input_app[sample_id_app]
                        target_app[b] = self.labels_app[sample_id_app]
                    else:
                        sample_id_app = sample_ids_app.pop()    # 出栈
                        x_app[b][0] = self.user_input_app[sample_id_app]
                        x_app[b][1] = self.item_input_app[sample_id_app]
                        target_app[b] = self.labels_app[sample_id_app]
                keys = [self.input_app, self.target_app]
                values = [x_app, target_app]
                _, loss_app, pred_app, self.step = self.sess.run([self.optim_app,
                                                    self.loss_app_only,
                                                    self.pred_app_only,
                                                    self.global_step],
                                                    feed_dict={
                                                        k:v for k,v in zip(keys, values)
                                                    })
                cost_app += np.sum(loss_app)
                cost_total += np.sum(loss_app)

        #  训练公共块 joint training on both datasets after single training on the bigger one 先单一训练后联合训练
        for _ in batches_join:

            target_app.fill(0)
            target_news.fill(0)
            for b in xrange(self.batch_size):
                if not sample_ids_app:  # if app used up
                    sample_id_app = random.randrange(0, num_examples_app)   # 打乱顺序 重新选
                    x_app[b][0] = self.user_input_app[sample_id_app]
                    x_app[b][1] = self.item_input_app[sample_id_app]
                    target_app[b] = self.labels_app[sample_id_app]
                else:
                    sample_id_app = sample_ids_app.pop()
                    x_app[b][0] = self.user_input_app[sample_id_app]
                    x_app[b][1] = self.item_input_app[sample_id_app]
                    target_app[b] = self.labels_app[sample_id_app]
                if not sample_ids_news:  # if news used up
                    sample_id_news = random.randrange(0, num_examples_news)
                    x_news[b][0] = self.user_input_news[sample_id_news]
                    x_news[b][1] = self.item_input_news[sample_id_news]
                    target_news[b] = self.labels_news[sample_id_news]
                else:
                    sample_id_news = sample_ids_news.pop()
                    x_news[b][0] = self.user_input_news[sample_id_news]
                    x_news[b][1] = self.item_input_news[sample_id_news]
                    target_news[b] = self.labels_news[sample_id_news]

            keys = [self.input_app, self.input_news, self.target_app, self.target_news]
            values = [x_app, x_news, target_app, target_news]
            _, loss, loss_app, loss_news, pred_app, pred_news, self.step = self.sess.run([self.optim_joint,
                                                self.loss_joint,
                                                self.loss_app_joint,
                                                self.loss_news_joint,
                                                self.pred_app_joint,
                                                self.pred_news_joint,
                                                self.global_step],
                                                feed_dict={
                                                    k:v for k,v in zip(keys, values)
                                                })
            cost_joint += np.sum(loss)
            cost_total += np.sum(loss)
            cost_app += np.sum(loss_app)
            cost_news += np.sum(loss_news)
        return [cost_total/num_batches_max/self.batch_size,cost_joint/num_batches_min/self.batch_size,
                cost_app/num_batches_app/self.batch_size,cost_news/num_batches_news/self.batch_size]

    def run(self):
        print('开始训练')
        self.get_test_instances_app()  # only need to get once
        self.get_test_instances_news()  # only need to get once

        # 开始迭代训练，同时验证
        start_time = time.time()
        for idx in xrange(self.nepoch):     # 开始迭代
            start = time.time()
            train_loss_total, train_loss_joint, train_loss_app, train_loss_news = self.train_model()
            train_time = time.time() - start
            print('=' * 80)
            print('迭代', idx, '： train time', train_time, 'train loss total', train_loss_total)
            # 测试    调用测试函数
            # start = time.time()
            self.rmse_app = self.valid_model_app()
            self.rmse_news = self.valid_model_news()

            if self.rmse_news < self.bestrmse_news: # 如果本次结果比之前都好
                self.bestrmse_news = self.rmse_news
                self.bestrmse_epoch_news = idx
            if self.rmse_app < self.bestrmse_app:
                self.bestrmse_app = self.rmse_app
                self.bestrmse_epoch_app = idx
            # valid_loss = (valid_loss_app + valid_loss_news) / 2
            # valid_time = time.time() - start
            # print(idx, 'test time', valid_time)

    # 利用测试集进行测试，输入所有验证集的用户、物品对 和真实评分直接计算rmse
    def valid_model_app(self):
        num_test_examples = len(self.test_labels_app)   # 测试物品总数
        num_test_batches = math.ceil(num_test_examples / self.batch_size)
        cost = 0
        x = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)  # user,item
        target = np.zeros([self.batch_size, self.class_size])

        rmse_app = 0
        sample_id = 0
        test_preds = []
        for current_batch in xrange(num_test_batches):  # 遍历测试集的每一个快,测试集中的数据也是按照块计算的
            for b in xrange(self.batch_size):   # 获取一个块的数据~~~~~~~~ 少的部分用第一个补齐 每一个块b=1,2...128
                if sample_id >= len(self.test_labels_app):
                    x[b][0] = self.test_user_input_app[0]
                    x[b][1] = self.test_item_input_app[0]
                    target[b] = self.test_labels_app[0]
                else :  # fill this batch; not be used when compute test metrics
                    x[b][0] = self.test_user_input_app[sample_id]
                    x[b][1] = self.test_item_input_app[sample_id]
                    target[b] = self.test_labels_app[sample_id]
                sample_id += 1  # 递增实现测试集数据的遍历

            keys = [self.input_app, self.target_app]
            values = [x, target]
            loss, pred = self.sess.run([self.loss_app_only, self.pred_app_only],
                                    feed_dict={
                                        k:v for k,v in zip(keys, values)
                                    })
            cost += np.sum(loss)
            test_preds.extend(pred)     # 得到预测结果
        # evaluation
            rmse_app = rmse_app + (np.mean(np.square(pred - target)))    # 当前块的rmse+前面块的rmse
        print('app test rmse: ', np.sqrt(rmse_app/num_test_batches))
        return np.sqrt(rmse_app/num_test_batches) # rmse_app/num_test_batches

    def valid_model_news(self):
        num_test_examples = len(self.test_labels_news)
        num_test_batches = math.ceil(num_test_examples / self.batch_size)
        cost = 0
        x = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.class_size])

        rmse_news = 0
        sample_id = 0
        test_preds = []
        for current_batch in xrange(num_test_batches):  # 遍历块
            target.fill(0)
            for b in xrange(self.batch_size):   # 遍历块内,获取一个块的数据x target
                if sample_id >= len(self.test_labels_news):  # fill this batch; not be used when compute test metrics
                    x[b][0] = self.test_user_input_news[0]
                    x[b][1] = self.test_item_input_news[0]
                    target[b] = self.test_labels_news[0]
                else:
                    x[b][0] = self.test_user_input_news[sample_id]
                    x[b][1] = self.test_item_input_news[sample_id]
                    target[b] = self.test_labels_news[sample_id]
                sample_id += 1

            keys = [self.input_news, self.target_news]
            values = [x, target]
            loss, pred = self.sess.run([self.loss_news_only, self.pred_news_only],
                                       feed_dict={
                                           k: v for k, v in zip(keys, values)
                                       })
            cost += np.sum(loss)
            test_preds.extend(pred)
            rmse_news = rmse_news + (np.mean(np.square(pred - target)))
        print('news test rmse', np.sqrt(rmse_news/num_test_batches))
        return np.sqrt(rmse_news/num_test_batches)   # rmse_news/num_test_batches