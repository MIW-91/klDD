# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:17:08 2020

@author: huang
"""


import os
import scipy.stats as stat
import math
import numpy as np
import pandas as pd
import random
import time
from sklearn.decomposition import PCA


def multi_dimensional_kl_divergence(x,y):
    
    mean1=np.mean(x,axis=1)
    cov1=np.cov(x)
    det_cov1=np.linalg.det(cov1)
    
    mean2=np.mean(y,axis=1)
    cov2=np.cov(y)
    inv_cov2=np.linalg.inv(cov2)
    det_cov2=np.linalg.det(cov2)
    
    trace_cov12=np.trace(np.dot(inv_cov2,cov1))
    mean_12=mean2-mean1
    kl=1/2*(np.log(det_cov2/det_cov1)-len(cov1)+trace_cov12+np.dot(np.dot(mean_12.transpose(),inv_cov2),mean_12))    
    normalize_kl=kl/(np.log(len(x.T)*len(y.T)))
    normalize_kl_correct=normalize_kl/np.sqrt(np.sum(np.square(mean1-mean2)))
    
    return normalize_kl,normalize_kl_correct

def pca_process(x):
    
    x=x.T
    pca = PCA(n_components=2)   #降到2维
    pca.fit(x)                  #训练
    new_x=pca.fit_transform(x)   #降维后的数据
    new_x=new_x.T    
    
    return new_x

path_open="/klPDM/datasets"

##control stage samples
ref=pd.read_csv(path_open+os.sep+"control_normal_liver_tissue.csv",index_col=0)
ref=ref.loc[~ref.index.duplicated(keep='first')]
data_genes=list(ref.index)


##tf-regulate-target
f=open(path_open+os.sep+"hg19_tf2targ.Up2kDown1k.txt")
network={}
flag=0
for p in f:
    flag+=1
    t=p.split()
    if flag==1:
        continue
    if t[0] not in data_genes:
        continue
    if t[0] not in network.keys():
        network[t[0]]=[]
    if t[1] not in network[t[0]] and t[1] in data_genes:
        network[t[0]].append(t[1])
f.close()

network_keys=list(network.keys())

fw=open(path_open+os.sep+"tf_target_database.txt","w")

for tf in network_keys:
    target_genes=network[tf]
    for gene in target_genes:
        fw.write(tf+"\t"+gene+"\n")
fw.close()

list(set().intersection())
tf_target=pd.read_table(path_open+os.sep+"hg19_tf2targ.Up2kDown1k.txt")

path_stages=path_open+os.sep+"stage_data"
stages=os.listdir(path_stages)

genes_score0={}
genes_score1={}
except_genes=[]  
nan_genes=[]  
for gene in network_keys:
    
    gene_sore0=[]
    gene_sore1=[]
    module_genes=[gene]+network[gene]
    for i in range(len(stages)-1):
        stage0=stages[i]
        stage1=stages[i+1]
        data=pd.read_csv(path_stages+os.sep+stage1,index_col=0)
        data=data.loc[~data.index.duplicated(keep='first')]
        
        ref=pd.read_csv(path_stages+os.sep+stage0,index_col=0)
        ref=ref.loc[~ref.index.duplicated(keep='first')]
                
        control_module=pd.DataFrame(ref,index=module_genes)
        control_module=control_module.dropna(axis=0,how="any")
        new_control_module = pca_process( control_module)

        stage_module=pd.DataFrame(data,index=module_genes)
        stage_module=stage_module.dropna(axis=0,how="any")
        new_stage_module = pca_process(stage_module)

        try:
            score0=1/2*(multi_dimensional_kl_divergence(new_control_module, new_stage_module)[0]+multi_dimensional_kl_divergence(new_stage_module,new_control_module)[0])
            gene_sore0.append(score0)
        except:
            except_genes.append(gene)
    
    if len(gene_sore0) < 7:
        continue
    
    if True in np.isnan(gene_sore0):
        nan_genes.append(gene)
        continue
    genes_score0[gene]=gene_sore0
    genes_score1[gene]=gene_sore1

df0=pd.DataFrame(genes_score0).T

df0.to_csv(path_open+os.sep+"genes_module_score_normlization_tf0_next.csv")



df0=pd.read_csv("genes_module_score_normlization_tf0_next.csv",index_col=0)
import matplotlib.pyplot as plt 

x=[i for i in range(1,8)]
for i in range(len(df0)):
    y=df0.iloc[i]
    plt.plot(x, y,linewidth=2)
plt.xlabel("Sampling time point")
plt.ylabel("KL-score of Regulon")
plt.savefig("score of regulon.pdf")
plt.show()

