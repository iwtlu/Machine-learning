# Machine-learning
2021年11月机器学习课程大作业
### CoNet-old
CoNet-old为论文CoNet: Collaborative Cross Networks for Cross-Domain Recommendation原文中原代码。
### CoNet-new
CoNet-new为本次大作业所改进的源代码。
CoNet_variantdata为改进后的CoNet模型：
+ 将CoNet改为预测评分，添加AntoEncoder自编码器提取特征
+ 将原始输入变为经过自编码预先训练好的用户/物品特征矩阵
### DataSet：data
选取了Amazon平台所提供的ratings CDS and Vinyl. csv和ratings Movies and TV. csv数据集，数据集中包括三项数据：User_ID（用户ID）、Item_ID（物品ID）、用户对物品的评分和Unix时间。
+ 删除用户评分条目数<5的所有用户
+ 按照8：2的比例划分训练集和测试集
+ 每个用户选择1个正例，99个负例组成测试集，余下的正例物品作为训练集
