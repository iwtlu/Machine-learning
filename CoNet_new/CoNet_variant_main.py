# 变种实验主函数
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import pprint
import tensorflow as tf
import time
from collections import defaultdict
from CoNet_variant import MTL
import json
import random
import math
import numpy as np
pp = pprint.PrettyPrinter()

def main(_):
    paras_setting = {
        'edim_u': 200,   # 用户特征维度
        'edim_f': 150,  # 编码/解码层特征维度
        'layers': [150,100,50],  # layers[0] must equal to edim_u + edim_v
        'batch_size': 128,  # "batch size to use during training [128,256,512,]"
        'nepoch': 200,  # "number of epoch to use during training [80]"
        'init_lr': 0.001,  # "initial learning rate [0.01]"
        'init_std': 0.01,  # "weight initialization std [0.05]"
        'max_grad_norm': 10,  # "clip gradients to this norm [50]"
        'negRatio': 1,  # "negative sampling ratio [5]"
        'cross_layers': 2,  # cross between 1st & 2nd, and 2nd & 3rd layers
        #'merge_ui': 0,  # "merge embeddings of user and item: 0-add, 1-mult [1], 2-concat"
        'activation': 'relu',  # "0:relu, 1:tanh, 2:softmax"
        'learner': 'adam',  # {adam, rmsprop, adagrad, sgd}
        'objective': 'mse',  # 0:cross, 1: hinge, 2:log
        #'carry_trans_alpha': [0.5, 0.5],  # weight of carry/copy gate
        'data_dir': 'data/amazon2/',  # "data directory [../data]"
        'data_name_app': 'movie',  # "user-info", "data state [user-info]"
        'data_name_news': 'music',  # "user-info", "data state [user-info]"
        'weights_app_news': [1, 1],  # weights of each task [0.8,0.2], [0.5,0.5], [1,1]
        'checkpoint_dir': 'checkpoints',  # "checkpoints", "checkpoint directory [checkpoints]"
        'show': True,  # "print progress [True]"
        #'isDebug': True,  # "isDebug mode [True]"
        'isDebug': False,  # "isDebug mode [True]"
        'isOneBatch': False,  # "isOneBatch mode for quickly run through [True]"
    }
    # setenv CUDA_VISIBLE_DEVICES 1
    isRandomSearch = False

    if not isRandomSearch:  # 默认执行
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.6  # 占用显存比例
        start_time = time.time()
        with tf.Session() as sess:
            model = MTL(paras_setting, sess)
            model.build_model()     # 创建模型
            model.run()     # 调用run()
            metrics = {
                'bestrmse_app': model.bestrmse_app,
                'bestrmse_epoch_app': model.bestrmse_epoch_app,
                'bestrmse_news': model.bestrmse_news,
                'bestrmse_epoch_news': model.bestrmse_epoch_news,
            }
            print('=' * 80)
            pp.pprint(metrics)
        print('total time {:.2f}m'.format((time.time() - start_time)/60))

if __name__ == '__main__':
    tf.app.run()    # 执行main函数
