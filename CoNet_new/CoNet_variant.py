# CoNet的变种实验
'''
直接使用用户特征进行迁移，即去掉物品特征部分
采用评分预测
'''
import numpy as np
import tensorflow as tf
from past.builtins import xrange
from utils import Dataset
from collections import defaultdict
import time
import sys
import math
import random



class MTL(object):
    def __init__(self, config, sess):
        t1 = time.time()
        # data corpus and load data
        self.data_dir = config['data_dir']

        # 1) task: app rec
        self.data_name_app = config['data_name_app']
        dataset_app = Dataset(self.data_dir + self.data_name_app)
        self.nUsers_app, self.nItems_app = dataset_app.nUsers, dataset_app.nItems
        self.train_app, self.test_app, self.embed_app = dataset_app.trainMatrix, dataset_app.testMatrix, dataset_app.embedMatrix
        print("----Load app data done [%.1f s]. #user=%d, #item=%d"
          %(time.time()-t1, self.nUsers_app, self.nItems_app))

        # 2) task: news rec
        self.data_name_news = config['data_name_news']
        dataset_news = Dataset(self.data_dir + self.data_name_news)
        self.nUsers_news, self.nItems_news = dataset_news.nUsers, dataset_news.nItems
        self.train_news, self.test_news, self.embed_news = dataset_news.trainMatrix, dataset_news.testMatrix, dataset_news.embedMatrix # 类似矩阵 但比矩阵方便 嘿嘿嘿
        print("----Load news data done [%.1f s]. #user=%d, #item=%d"
          %(time.time()-t1, self.nUsers_news, self.nItems_news))

        if self.nUsers_app != self.nUsers_news:     # 要求总用户数必须相同 在这里应该
            print('nUsers_app != nUsers_news. However, they should be shared. exit...')
            sys.exit(0)
        self.nUsers = self.nUsers_news

        # hyper-parameters
        self.init_std = config['init_std']
        self.batch_size = config['batch_size']
        self.nepoch = config['nepoch']
        self.layers = config['layers']
        self.edim_u = config['edim_u']
        self.edim_f = config['edim_f']
        self.nhop = len(self.layers)
        self.max_grad_norm = config['max_grad_norm']
        self.activation = config['activation']
        self.learner = config['learner']
        self.objective = config['objective']

        # save and restore
        self.show = config['show']
        self.checkpoint_dir = config['checkpoint_dir']

        # 神经网络输入层、输出层---空层
        self.UIin_app = tf.placeholder(tf.float32, [None, self.edim_u], name="input_app")   # 用户特征输入
        self.UIin_news = tf.placeholder(tf.float32, [None, self.edim_u], name="input_app")
        self.target_app = tf.placeholder(tf.float32, [None, self.nItems_app], name="target")
        self.target_news = tf.placeholder(tf.float32, [None, self.nItems_news], name="target")

        self.lr = None
        self.init_lr = config['init_lr']
        self.current_lr = config['init_lr']
        self.loss_joint = None
        self.loss_app_joint = None
        self.loss_news_joint = None
        self.optim_joint = None
        self.step = None
        self.sess = sess
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
        self.num_batch = self.nUsers // self.batch_size     # 总块数 按用户分块 一次输入多个用户的特征矩阵

    # 共享参数
    def build_memory_shared(self):
        ## ------- parameters: shared-------  ##
        # match the dimensions... 跨域迁移模块
        self.shared_Hs = defaultdict(object)    # 设置对象 包含多层
        for h in xrange(1, self.cross_layers+1):  # only cross between h=1 2 64*32---32*16
            self.shared_Hs[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))

    # 一个域的私有参数
    def build_memory_app_specific(self):
        ## ------- parameters: app specific -------  ##
        # 1. embedding matrices for input <user, app>, <user, news>: shared user embedding matrix
        # 自编码的特征提取层 权重 偏置
        self.encode_w_app = tf.Variable(tf.random_normal([self.edim_u, self.edim_f], stddev=self.init_std))
        self.encode_b_app = tf.Variable(tf.random_normal([self.edim_f], stddev=self.init_std))

        # 2. weights & biases for hidden layers: the input to hidden layers are the merged embedding app域的权重、偏置
        self.weights_app = defaultdict(object)
        self.biases_app = defaultdict(object)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            self.weights_app[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))
            self.biases_app[h] = tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std))

        # 3. output layer: weight and bias 解码层的权重 偏置
        self.h_app = tf.Variable(tf.random_normal([self.layers[-1], self.edim_f], stddev=self.init_std))
        self.b_app = tf.Variable(tf.random_normal([self.edim_f], stddev=self.init_std))
        self.out_w_app = tf.Variable(tf.random_normal([self.edim_f, self.nItems_app], stddev=self.init_std))
        self.out_b_app = tf.Variable(tf.random_normal([self.nItems_app], stddev=self.init_std))

    def build_memory_news_specific(self):
        ## ------- parameters: news specific -------  ##
        # 1. embedding matrices for input <user, app>, <user, news>: shared user embedding matrix
        self.encode_w_news = tf.Variable(tf.random_normal([self.edim_u, self.edim_f], stddev=self.init_std))
        self.encode_b_news = tf.Variable(tf.random_normal([self.edim_f], stddev=self.init_std))
        # 2. weights & biases for hidden layers: the input to hidden layers are the merged embedding
        self.weights_news = defaultdict(object)
        self.biases_news = defaultdict(object)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            self.weights_news[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))
            self.biases_news[h] = tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std))
        # 3. output layer: weight and bias
        self.h_news = tf.Variable(tf.random_normal([self.layers[-1], self.edim_f], stddev=self.init_std))
        self.b_news = tf.Variable(tf.random_normal([self.edim_f], stddev=self.init_std))
        self.out_w_news = tf.Variable(tf.random_normal([self.edim_f, self.nItems_news], stddev=self.init_std))
        self.out_b_news = tf.Variable(tf.random_normal([self.nItems_news], stddev=self.init_std))

    # 模型搭建
    def build_model_joint_training(self):
        # 1、输入特征再提取
        self.layer_h_apps = defaultdict(object)
        layer_h_app = tf.nn.relu(tf.add(tf.matmul(self.UIin_app, self.encode_w_app), self.encode_b_app))
        self.layer_h_apps[0] = layer_h_app

        self.layer_h_newss = defaultdict(object)
        layer_h_news = tf.nn.relu(tf.add(tf.matmul(self.UIin_news, self.encode_w_news), self.encode_b_news))
        self.layer_h_newss[0] = layer_h_news

        # 2、跨域层
        for h in xrange(1, self.nhop):
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

        # 3. 最后 解码层
        self.decode_app = tf.nn.relu(tf.matmul(layer_h_app, self.h_app) + self.b_app)
        self.z_app_joint = tf.matmul(self.decode_app, self.out_w_app) + self.out_b_app
        self.pred_app_joint = tf.identity(self.z_app_joint)

        self.decode_news = tf.nn.relu(tf.matmul(layer_h_news, self.h_news) + self.b_news)
        self.z_news_joint = tf.matmul(self.decode_news, self.out_w_news) + self.out_b_news
        self.pred_news_joint = tf.identity(self.z_news_joint)

        # 只和原始有评分的计算损失函数
        self.pred_app_joint = tf.multiply(self.pred_app_joint, tf.sign(self.target_app))
        self.pred_news_joint = tf.multiply(self.pred_news_joint, tf.sign(self.target_news))

        ## ------- loss and optimization ------- ##
        # 1、app的损失函数
        if self.objective == 'cross':
            self.loss_app_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_app_joint, labels=self.target_app)
        elif self.objective == 'log':
            self.loss_app_joint = tf.losses.log_loss(predictions=self.pred_app_joint, labels=self.target_app)
        elif self.objective == 'mse':   # mse损失函数
            # 只和原始有评分的比较
            self.loss_app_joint = tf.losses.mean_squared_error(predictions=self.pred_app_joint, labels=self.target_app)
        else:
            self.loss_app_joint = tf.losses.hinge_loss(logits=self.z_app_joint, labels=self.target_app)

        # 2、news的损失函数
        if self.objective == 'cross':
            self.loss_news_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_news_joint, labels=self.target_news)
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

        params = [self.encode_w_app, self.encode_b_app, self.h_app, self.b_app, self.out_w_app, self.out_b_app,\
                  self.encode_w_news, self.encode_b_news, self.h_news, self.b_news, self.out_w_news, self.out_b_news]
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
            self.optim_joint = self.opt_joint.apply_gradients(clipped_grads_and_vars)

    # 搭建神经网络模型
    def build_model(self):
        print('开始搭建神经网络模型')
        self.global_step = tf.Variable(0, name="global_step")

        self.build_memory_shared()  # 共享参数
        self.build_memory_app_specific()    # 域1特定参数
        self.build_memory_news_specific()   # 域2特定参数

        self.build_model_joint_training()
        tf.global_variables_initializer().run()

    # 一次完整的训练
    def train_model(self, epoch):
        n_batches = 0
        total_loss = 0
        random_user_idx = np.random.permutation(self.nUsers)  # 打乱用户顺序
        start_time = time.time()

        for i in range(self.num_batch):     # i=块数
            if i == self.num_batch - 1:
                batch_idx = random_user_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_idx = random_user_idx[i * self.batch_size: (i + 1) * self.batch_size]
            feed_config = {
                self.UIin_app:self.embed_app[batch_idx],
                self.UIin_news:self.embed_news[batch_idx],
                self.target_app:self.train_app[batch_idx],
                self.target_news:self.train_news[batch_idx],
            }
            _, batch_loss \
                = self.sess.run([self.optim_joint, self.loss_joint], feed_dict=feed_config)
            n_batches += 1
            total_loss += batch_loss

        print("=" * 80)
        print("Training Epoch {0}: [Loss] {1}, [TIME] {2}".format(epoch, total_loss / n_batches, time.time() - start_time))
        return total_loss / n_batches


    def run(self):
        print('开始训练')
        # 迭代训练，同时验证
        for idx in xrange(self.nepoch):     # 开始迭代
            train_loss_total = self.train_model(idx)
            self.rmse_app, self.rmse_news = self.valid_model(idx)

            if self.rmse_news < self.bestrmse_news:  # 如果本次结果比之前都好
                self.bestrmse_news = self.rmse_news
                self.bestrmse_epoch_news = idx
            if self.rmse_app < self.bestrmse_app:
                self.bestrmse_app = self.rmse_app
                self.bestrmse_epoch_app = idx

    # 测试
    def valid_model(self, epoch):
        start_time = time.time()
        loss, pred_app, pred_news \
            = self.sess.run([self.loss_joint, self.pred_app_joint, self.pred_news_joint],
                               feed_dict={
                                   self.UIin_app: self.embed_app,
                                   self.UIin_news: self.embed_news,
                                   self.target_app: self.test_app,
                                   self.target_news: self.test_news,
                               })
        pred_app, pred_news = pred_app.clip(min=1, max=5), pred_news.clip(min=1, max=5)
        test_pred_app = pred_app[self.test_app.nonzero()]
        truth_app = self.test_app[self.test_app.nonzero()]
        rmse_app = np.sqrt(np.mean(np.square(truth_app - test_pred_app)))

        test_pred_news = pred_news[self.test_news.nonzero()]
        truth_news = self.test_news[self.test_news.nonzero()]
        rmse_news = np.sqrt(np.mean(np.square(truth_news - test_pred_news)))
        print("Testing Epoch {0} : [LOSS] {1} ,[RMSE_app] {2} , [RMSE_news] {3}, [TIME] {4}".\
              format(epoch, loss, rmse_app, rmse_news, (time.time() - start_time)))
        return rmse_app, rmse_news

