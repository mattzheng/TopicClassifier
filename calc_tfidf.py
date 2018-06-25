# -*- coding: utf-8 -*-

from tqdm import tqdm
import pandas as pd
import math
import pickle

def docs(w, D):
    '''
        w,词;
        D,整个文档，分好词的，譬如[('你','好'),('我','们')....]
        计算含有w词的文档数量
    '''
    c = 0
    for d in D:
        if w in d:
            c = c + 1;
    return c

def Flatten(texts):
    diction = []
    for txt in tqdm(texts):
        diction.extend(txt)
    return diction

def get_idf(jieba_content):
    '''
    作用：生成、计算idf
    参考：http://www.voidcn.com/article/p-mhebqvic-qq.html
    输入：分行的文本内容;
    输出：dataframe,分别有:words /  tf / idf  三列
    '''
    print('start ...')
    diction = pd.DataFrame( list(set(  Flatten(jieba_content) ) ) , columns = ['words'])
    print('Calculate IDF ...')
    #diction['tf'] =  list(map(lambda x : docs(x,jieba_content)  , diction['words'] ))
    #
    tf = []
    for i in tqdm(diction['words'] ):
        tf.append(docs(i,jieba_content) )
    diction['tf'] =  tf
    #
    n = len(jieba_content)
    diction['idf'] = [math.log(i) for i in   n*1.0 / (diction['tf'] + 1 ) ]
    return diction


if __name__ == '__main__':
    # 加载训练语料
    toutiao_data = pd.read_csv('./toutiao_data.csv',encoding = 'utf-8')
    
    # 变成list
    # 类似：[['1','2'],['3','4']]
    texts = []
    for i,j in tqdm(zip(toutiao_data['keyword_split'],toutiao_data['label_split']),total = len(toutiao_data)):
       texts.append(eval(i) + eval(j)) 
    
    # 计算
    diction = get_idf(texts[:1000])
    
    # save
    pickle.dump(diction,open('./diction.pkl','wb')   )  
    #diction = pickle.load(open('./diction.pkl', 'rb'))
