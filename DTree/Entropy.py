from math import log
import operator

"""
函数说明: 创建测试数据集
Parameters: 无
Returns:
    DataSet: 数据集
    Labels: 分类属性值
Modify:
    2020-06-16
"""

def  CreateDataSet():

    # 数据集大小: 15条数据，属性维度为5
    DataSet=[[0, 0, 0, 0, 'no'],
             [0, 0, 0, 1, 'no'],
             [0, 1, 0, 1, 'yes'],
             [0, 1, 1, 0, 'yes'],
             [0, 0, 0, 0, 'no'],
             [1, 0, 0, 0, 'no'],
             [1, 0, 0, 1, 'no'],
             [1, 1, 1, 1, 'yes'],
             [1, 0, 1, 2, 'yes'],
             [1, 0, 1, 2, 'yes'],
             [2, 0, 1, 2, 'yes'],
             [2, 0, 1, 1, 'yes'],
             [2, 1, 0, 1, 'yes'],
             [2, 1, 0, 2, 'yes'],
             [2, 0, 0, 0, 'no']]

    #分类标签:
    Labels=['年龄','工作','房子','信贷问题']
    return DataSet,Labels

"""
函数说明: 计算给定数据集的经验熵
Parameters: 
    DataSet: 数据集
Return:
    Entropy: 经验熵
Modify:
    2020-06-16
"""

def CalculateEntropy(DataSet):

    #返回数据集的个数
    NumData=len(DataSet)
    #保存每个标签出现次数的字典
    LabelsCount={}
    #对每组特征向量进行统计
    for Feature in DataSet:
        CurrentLabels=Feature[-1]                          #提取标签信息
        #添加Labels(标签),对不存在的key赋值，就是添加键值对。
        if CurrentLabels not in LabelsCount.keys():
            LabelsCount[CurrentLabels]=0                   #将没有添加入统计次数的标签放入
        LabelsCount[CurrentLabels]+=1                      #Label计数

    Entropy=0.0                                            #熵

    #计算经验熵
    for key in LabelsCount:
        Prob=float(LabelsCount[key])/NumData
        Entropy-=Prob*log(Prob,2)
    return Entropy

"""
函数说明: 按照给定特征划分数据集
Parameter: 
    DataSet：待划分的数据集
    Axis: 化分数据集的特征
    Value: 需要返回的特征的值
Return:
    SubDataSet: 子数据集
Modify:
    2020-06-23
"""
def SplitDataSet(DataSet,Axis,Value):
    SubDataSet=[]
    for Data in DataSet:
        if Data[Axis]==Value:
            #去掉axis属性
            ReduceFeatVec=Data[:Axis]
            ReduceFeatVec.extend(Data[Axis+1:])
            SubDataSet.append(ReduceFeatVec)
    return SubDataSet

"""
函数说明: 计算各属性划分的分支节点的信息熵，及属性的信息增益
Parameter: 
    DataSet：数据集
Return:
    NodeEntropy: 分支节点的信息熵
    Gain: 信息增益
Modify:
    2020-06-16
"""
def InfoGain(DataSet):
    #特征属性数量
    NumAttribute=len(DataSet[0])-1
    #计算数据的熵
    BaseEntropy=CalculateEntropy(DataSet)
    #最优信息增益
    BestInfoGain=0.0
    #最优特征的索引值
    BestFeature=-1

    #遍历所有特征
    for i in range(NumAttribute):
        # 获取DataSet的第i个所有特征
        AttrList=[example[i] for example in DataSet]
        # 创建set集合{}，元素不可重复
        UniqueAttr=set(AttrList)
        # 经验条件熵
        ConEntropy=0.0
        # 计算信息增益
        for value in UniqueAttr:
            # SubDataSet 划分后的子集
            SubDataSet=SplitDataSet(DataSet,i,value)
            # 计算子集的概率
            Prob=len(SubDataSet)/float(len(DataSet))
            # 计算经验条件熵
            ConEntropy += Prob * CalculateEntropy(SubDataSet)
        # 信息增益
        InfoGain=BaseEntropy - ConEntropy
        # 打印第i个特征的信息增益
        print("第%d个特征的信息增益是%.3f"%(i,InfoGain))
        # 计算最优信息增益
        if(InfoGain > BestInfoGain):
            #更新最优信息增益
            BestInfoGain = InfoGain
            BestFeature = i
        #返回最优特征索引值
    return BestFeature

"""
函数说明: 统计ClassList中出现次数最多的类标签
Parameter: 
    ClassList: 类标签列表
Return:
    SortedClassCount[0][0]: 出现次数最多的元素
Modify:
    2020-06-23
"""
def MajorLabel(ClassList):
    ClassCount={}
    # 统计ClassList每个元素出现的次数
    for vote in ClassList:
        if vote not in ClassCount.keys():
            ClassCount[vote]=0
        ClassCount[vote]+=1
    SortedClassCount=sorted(ClassCount.items(),key=operator.itemgetter(1),reverse=True)
    return SortedClassCount[0][0]

"""
函数说明: 基于ID3创建决策树
Parameter: 
    DataSet：待划分的数据集
    Labels: 分类属性标签
    FeatLabels: 存储选择最优标签
Return:
    DecisionTree: 决策树
Modify:
    2020-06-23
"""

def CreateTree(DataSet,Labels,FeatLabels):
    # 取分类标签
    ClassList= [example[-1] for example in DataSet]
    # 如果类别相同，停止划分
    if ClassList.count(ClassList[0])==len(ClassList):
        return ClassList[0]
    # 特征集为空，遍历完所有特征时返回出现次数最多的类标签.
    if len(DataSet[0])==1:
        return MajorLabel(ClassList)
    # 选择最优特征、标签
    BestFeature= InfoGain(DataSet)
    BestLabel= Labels[BestFeature]
    FeatLabels.append(BestLabel)
    # 利用最优特征标签构建决策树
    DecisionTree={BestLabel:{}}
    # 删除已经使用过的特征标签
    del(Labels[BestFeature])
    # 得到训练集中所有最优特征的属性值
    FeatValue=[example[BestFeature] for example in DataSet]
    #去掉重复的属性值
    UniqueFeatValue=set(FeatValue)
    #遍历特征，创建决策树
    for subValue in  UniqueFeatValue:
        DecisionTree[BestLabel][subValue]=CreateTree(SplitDataSet(DataSet,BestFeature,subValue),Labels,FeatLabels)
    return DecisionTree

#main函数
if __name__=='__main__':
    DataSet,Labels=CreateDataSet()
    print(DataSet)
    # 根节点的信息熵
    print(CalculateEntropy(DataSet))
    # 最优索引值
    # print("最优索引值：" +str(InfoGain(DataSet)))
    FeatLabels=[]
    DecisionTree=CreateTree(DataSet,Labels,FeatLabels)
    print(DecisionTree)