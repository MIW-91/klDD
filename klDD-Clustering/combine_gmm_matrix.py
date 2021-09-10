# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 08:50:34 2020

@author: huang
"""

import os
import numpy as np
import multiprocessing

def work(num,open_path,cell_list):
    
    print("subprocessing:",num)
    cell=cell_list[num]
    
    f=open(open_path+os.sep+cell+os.sep+cell+".csv")
    flag=0
    for p in f:
        flag+=1
        t=p.split(",")
        if flag==1:
            samples=t[:]
            samples= [i for i in samples if i != '']
    f.close()
    


    genes=os.listdir(open_path+os.sep+cell+os.sep+"significant_gene_js_matrix")
            
    sample_matrix=np.zeros((len(samples),len(samples)))
    for gene in genes:
        gene_matrix=np.loadtxt(open_path+os.sep+cell+os.sep+"significant_gene_js_matrix"+os.sep+gene)
        sample_matrix=np.add(sample_matrix,gene_matrix)
    np.savetxt(open_path+os.sep+cell+os.sep+"sample_matrix_1time.txt",sample_matrix,fmt="%.0d")


if __name__ == "__main__":
    
    open_path="/gmm-single-cell/datasets"
    cell_list=os.listdir(open_path)   

    pool=multiprocessing.Pool(4)

    for num in range(len(cell_list)):
        pool.apply_async(work,args=(num,open_path,cell_list,))
    print("waiting for all subprocessing done")
    pool.close()
    pool.join()