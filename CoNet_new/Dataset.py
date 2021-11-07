# 根据路径加载数据 数据处理 train valid nega
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):

    def __init__(self, path):
        self.trainMatrix = self.load_rating_file_as_matrix(path + "_train.csv")  # 训练集
        self.testRatings = self.load_rating_file_as_matrix(path + "_test.csv")
        # self.testRatings = self.load_rating_file_as_list(path + ".valid.rating")    # 测试集
        # self.testNegatives = self.load_negative_file(path + ".neg.valid.rating")
        # assert len(self.testRatings) == len(self.testNegatives)
        
        # self.num_users, self.num_items = self.trainMatrix.shape
        nUsers_train, nItems_train = self.trainMatrix.shape
        nUsers_test, nItems_test = self.testRatings.shape
        self.nUsers, self.nItems = max(nUsers_train, nUsers_test), max(nItems_train, nItems_test)
        U = self.read_matrix(path + '_embed200.csv')
        V = self.read_matrix(path + '_item_embed200.csv')

        # 存在一个问题 用户1的物品嵌入矩阵在第0行 在矩阵中是0~n-1
        # 实际输入的是n
        # 解决方法 在矩阵第一行插入0，使得用户1在矩阵的第1行
        self.U = np.insert(U, 0, 0, axis=0)
        self.V = np.insert(V, 0, 0, axis=0)

    def load_rating_file_as_matrix(self, filename):
        # print('read {}...'.format(filename))
        # Get number of users and items 获取用户数  物品数
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)

                line = f.readline()
        # print('maxUserId = {}, maxItemId = {}'.format(num_users, num_items))

        # Construct matrix 构造矩阵
        users_set = set()
        items_set = set()
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)   # 初始化 并非矩阵，只记录评分非0的数据
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                mat[user, item] = rating
                users_set.add(user)
                items_set.add(item)
                line = f.readline()
        print('#(user,item,feed)={},{},{}'.format(len(users_set), len(items_set), mat.nnz))
        return mat

    # 读取特征矩阵
    def read_matrix(self, path):
        ratings = pd.read_csv(path, encoding="utf-8", header=None)
        ratings = ratings.iloc[:, :].values
        return ratings

    # 读取数据集
    def read_datas(self, path, num_users, num_items):
        fp = open(path)
        ratings = np.zeros((num_users, num_items))
        lines = fp.readlines()
        for line in lines:
            user, item, rating = line.split(',')
            user_idx = int(user) - 1
            item_idx = int(item) - 1
            ratings[user_idx, item_idx] = int(float(rating))
        return ratings


    def load_rating_file_as_list(self, filename):   # 返回用户物品对 组成的列表
        print('read {}...'.format(filename))
        users_set = set()
        items_set = set()
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                users_set.add(user)
                items_set.add(item)
                line = f.readline()
        print('#test_pos(user,item,feed)={},{},{}'.format(len(users_set), len(items_set), len(ratingList)))
        return ratingList
    
    def load_negative_file(self, filename):
        print('read {}...'.format(filename))
        negativeList = []
        items_set = set()
        nNegFeed = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[2:]:  # arr[0] = (user, pos_item)
                    item = int(x)
                    negatives.append(item)
                    items_set.add(item)
                    nNegFeed += 1
                negativeList.append(negatives)
                line = f.readline()
        print('#test_neg(item,feed)={},{}'.format(len(items_set), nNegFeed))
        return negativeList
    

