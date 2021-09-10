# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:20:23 2020

@author: huang
"""


import os
import math
import numpy as np
import multiprocessing
from sklearn.mixture import GaussianMixture

def js_divergence(x,y):

    x_mean=np.mean(x)
    x_std=np.std(x)
    y_mean=np.mean(y)
    y_std=np.std(y)
    z_mean=1/2*(x_mean+y_mean)
    z_std=1/2*math.sqrt(pow(x_std,2)+pow(y_std,2))
    kl1=1/2*((np.log(z_std)- np.log(x_std)+(1/2)*(pow(x_std,2)/ pow(z_std,2))+ (1/2)*pow(x_mean-z_mean,2)/ pow(z_std,2)-1/2))
    kl2=1/2*((np.log(z_std)- np.log(y_std)+(1/2)*(pow(y_std,2)/ pow(z_std,2))+ (1/2)*pow(y_mean-z_mean,2)/ pow(z_std,2)-1/2))
    js=kl1+kl2

    return js

def exact_mc_perm_test(xs, ys, nmc):
    n, t = len(xs), 0
    js = js_divergence(xs,ys)
    zs = np.concatenate([xs, ys])
    list_js=np.empty(nmc)
    for j in range(nmc):
        np.random.shuffle(zs)
        list_js[j]=js_divergence(zs[:n], zs[n:])
        t += js < js_divergence(zs[:n], zs[n:])
    p_value_js = t/nmc
    return  p_value_js


def work(num,path,cell_list):
    
    print("processing:",num)
    
    cell=cell_list[num]
    save_path1=path+os.sep+cell+os.sep+"significant_gene_js_matrix_1time"
    isexists=os.path.exists(save_path1)
     
    if not isexists:  
        os.makedirs(save_path1)
    
    data={}
    f=open(path+os.sep+cell+os.sep+cell+".csv")
    flag=0
    for p in f:
        flag+=1
        t=p.split(",")
        if flag==1:
            samples=t[:]
            samples= [i for i in samples if i != '']
        else:
            t_temp=[float(i) for i in t[1:]]
            # print( t_temp.count(0.0))
            if t_temp.count(0.0)/len(samples) < 0.95:
                print(t_temp.count(0.0)/len(samples))
                data[t[0]]=t_temp
    f.close()
    genes=list(data.keys())
    
    fw_labels=open(path+os.sep+cell+os.sep+"gene_label_1time.txt","w")
    fw_kl=open(path+os.sep+cell+os.sep+"gene_kl_1time.txt","w")
    
    for i in range(len(genes)):
        gene=genes[i]
        data_gene=data[gene]  
        data_gene=np.array(data_gene).reshape(len(data_gene),1)
        
        gmm=GaussianMixture(n_components=2).fit(data_gene)
        labels = gmm.predict(data_gene)
        
        group1=data_gene[labels==0]
        group2=data_gene[labels==1]
        
        if len(group1) > len(samples)*0.05  and len(group2) >  len(samples)*0.05:
            js= js_divergence(group1,group2)
            js_gene_pvalue= exact_mc_perm_test(group1, group2, 1000)
            if js_gene_pvalue < 0.01:
                labels_write=[str(m) for m in labels.tolist()]
                
                fw_labels.write(gene+"\t"+",".join(labels_write)+"\n")
                fw_kl.write(gene+"\t"+str(js)+"\t"+str(js_gene_pvalue)+"\n")
 
                similarity_matrix=np.zeros((len(labels),len(labels)))
                for j in range(len(labels)):
                    for k in range(len(labels)):
                        if labels[j]==labels[k]:
                            similarity_matrix[j][k]=1
                        else:
                            similarity_matrix[j][k]=0
        
                np.savetxt(save_path1+os.sep+gene+".txt",similarity_matrix,fmt="%.0d")                
            
    fw_labels.close()
    fw_kl.close()
 


if __name__ == "__main__":
    
    path="/sibcb1/chenluonanlab6/liuxiaoping/Temp-exp/team_work/gmm-single-cell/10-9"
    cell_list=os.listdir(path)

    pool=multiprocessing.Pool(2)

    for num in range(len(cell_list)):
        pool.apply_async(work,args=(num,path,cell_list,))
    print("waiting for all subprocessing done")
    pool.close()
    pool.join()
