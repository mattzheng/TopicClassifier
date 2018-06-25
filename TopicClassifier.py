# -*- coding: utf-8 -*-
import json
import pandas as pd
import jieba.posseg as pseg
import jieba.analyse
import itertools
from gensim import corpora, models, similarities
from tqdm import tqdm
import gensim
import jieba
import pickle
import copy 
import numpy as np


def not_nan(obj):
    return obj==obj

def getPseg(inputs,allowPOS = ['n','ns','vn','sx']):
    words = pseg.cut(inputs)
    words_pseg = {}
    for word, flag in words:
        words_pseg[word] = flag
    
    pseg_isnot = [i in allowPOS  for i in words_pseg.values()]
    return [j for i,j in enumerate(words_pseg.keys()) if pseg_isnot[i]]


#  整体主题密度
def totalTopic(topic_tag,topic_class,TopicDict):
    total_topic_list = copy.copy(topic_class)
    total_topic = [TopicDict[i]['topic'] for i in topic_tag]
    for totol in total_topic:
        for i in totol:
            total_topic_list[i] +=1
    return total_topic_list

# 单词粒度主题密度
def perTopic(topic_tag,topic_class,TopicDict):
    per_topic_list = copy.copy(topic_class)
    per_topic = [TopicDict[i]['topic_detail'] for i in topic_tag]
    for per in per_topic:
        for x,y in per.items():
            if y > 0:
                per_topic_list[x] += y
    return per_topic_list

# tfidf
def ShowTfidf(topic_tag,TopicDict):
    dict_tfidf = {}
    for i in topic_tag:
        dict_tfidf[i] = {}
        dict_tfidf[i]['tf'],dict_tfidf[i]['idf'],dict_tfidf[i]['tfidf'] = \
             TopicDict[i]['tf'],TopicDict[i]['idf'],TopicDict[i]['tf']*TopicDict[i]['idf']
    return dict_tfidf

# 排序算法
def TopN(per_topic_list,perc = 90):
    percentile = np.percentile(np.array(list(per_topic_list.values())),perc)
    means = np.mean(np.array(list(per_topic_list.values())))
    topSe = [[x,y,y/means] for x,y in per_topic_list.items() if y>percentile]
    toDict = {}
    for item in topSe:
        toDict[item[0]] = {}
        toDict[item[0]]['num'],toDict[item[0]]['degree'] = item[1],item[2]
    return toDict

# all

def TopicClassifier(sentense,TopicDict,topic_class,percs = 90,allowPOSs = ['topic']):
    topic_tag = getPseg(sentense,allowPOS = allowPOSs)
    total_topic_list = totalTopic(topic_tag,topic_class,TopicDict)  # 整体主题分类
    per_topic_list = perTopic(topic_tag,topic_class,TopicDict)  # 单个主题分类
    dict_topic = {'sentense':sentense,\
                  'totalTopic':TopN(total_topic_list,perc = percs),\
                  'perTopic':TopN(per_topic_list,perc = percs),\
                  'tfidf':ShowTfidf(topic_tag,TopicDict)}
    return dict_topic


if __name__ == '__main__':

    # 准备材料
    TopicDict = pickle.load(open('./output_0624.pkl', 'rb'))
    topic_material = eval(open('./topic_material.json', "r").read())
    # 自定义用户词典
    for word in tqdm(TopicDict.keys()):
        jieba.add_word(word, freq=10, tag='topic')
    # 分类
    sentense1 = '网易云音乐是一款专注于发现与分享的音乐产品,依托专业音乐人、DJ、好友推荐及社交功能,为用户打造全新的音乐生活。'
    sentense2 = '《创造101》终于收官了——经过昨晚（6月23日）的一夜鏖战，十一名女团人选最终确定：孟美岐、吴宣仪、杨超越、段奥娟、yamy、赖美云、紫宁、Sunnee（杨芸晴）、李紫婷、傅菁、徐梦洁。'
    sentense3 = '世界杯小组赛进入最后一轮，前2轮表现极其出色的C罗赢得了全世界的称赞，就连葡萄牙总统马塞洛-雷贝洛-德索萨也在同俄罗斯总统普京会面时，也不禁自夸：我们葡萄牙可是有C罗这种顶级巨星的。'
    TopicClassifier(sentense,TopicDict,topic_material['topic_class'],percs = 90,allowPOSs = ['topic'])
    
















