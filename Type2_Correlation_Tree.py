# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:14:25 2020

@author: Doowon
"""
#from numpy import * # donot recommended
import numpy as np
import scipy.stats as sttool
import math as mt
import pandas as pd
import sklearn.linear_model as lr

### Find all partitions of a list into k subsets
def sorted_k_partitions(seq, k):
    n = len(seq)
    groups = []  # a list of lists

    def generate_partitions(i):
        if i >= n:
            yield list(map(tuple, groups))
        else:
            if n - i > k - len(groups):
                for group in groups:
                    group.append(seq[i])
                    yield from generate_partitions(i + 1)
                    group.pop()

            if len(groups) < k:
                groups.append([seq[i]])
                yield from generate_partitions(i + 1)
                groups.pop()

    result = generate_partitions(0)

    # Sort the parts in each partition in shortlex order
    result = [sorted(ps, key = lambda p: (len(p), p)) for ps in result]
    # Sort partitions by the length of each part, then lexicographically.
    result = sorted(result, key = lambda ps: (*map(len, ps), ps))

    return result

### split the node by a continuous covariate
def binSplitDataSet(dataSet, feature, value):
    # the first [0] represents a row index meeting the condition
    right_node = np.nonzero(dataSet[:,feature]>value)[0]
    left_node = np.nonzero(dataSet[:,feature]<=value)[0]
    mat0 = dataSet[left_node,:]
    mat1 = dataSet[right_node,:]
    return mat0, mat1

### split the node by a categorical covariate
def cate_binSplitDataSet(df_original, dataSet, featIndex, splitVal1):
    df1 = df_original
    feat_name = df1.columns[featIndex]
    # convert array into data frame
    df = pd.DataFrame(dataSet, index=dataSet[:,0], columns=df1.columns)
    left_node = df.loc[df[feat_name].isin(list(splitVal1))]
    right_node = df.loc[~df[feat_name].isin(list(splitVal1))]
    mat0 = left_node.as_matrix()
    mat1 = right_node.as_matrix()
    return mat0, mat1

########## regression tree split function #############
def corLeaf(dataSet):
    if len(dataSet)==0:
        return 0
    return sttool.pearsonr(dataSet[:,0],dataSet[:,1])[0]
 
def Find_split_var_by_partialCor(df_original, dataSet):
    df = df_original
    if len(dataSet)==0:
        return 0
    num_obs, num_feature = np.shape(dataSet)
    
    #convert subset data array into panda dataframe
    df3 = pd.DataFrame(dataSet, index=dataSet[:,0], columns=df.columns)
    
    Edu_mapper= {0:'Other', 1:'College', 2:'Master', 3:'Ph.D.'}
    Edu_ordvar=df3["Edu"].replace(Edu_mapper)
    df3["Edu"]=list(pd.factorize(Edu_ordvar))[0]
        
    Gender_mapper= {0:'Female', 1:'Male'}
    Gender_ordvar=df3["Gender"].replace(Gender_mapper)
    df3["Gender"]=list(pd.factorize(Gender_ordvar))[0]
       
    ## df3 is encoded version of subdataset with DataFrame type
    ## df4 is converted from dataframe to array to 
    df4 = df3.as_matrix()
    appended_data =[]
    for featIndex in range(2,num_feature):
        y = pd.DataFrame(df4[:,0])
        x = pd.DataFrame(df4[:,1])
        z = pd.DataFrame(df4[:,featIndex])
        
        lm = lr.LinearRegression()
        model_zy = lm.fit(z,y)
        lm = lr.LinearRegression() ### has to be twice ##
        model_zx = lm.fit(z,x)
        
        residual_zy = y-model_zy.predict(z)
        residual_zx = x-model_zx.predict(z)
        
        res_zy = residual_zy.as_matrix().astype('float')[:,0]
        res_zx = residual_zx.as_matrix().astype('float')[:,0]
        parcor_xyz = sttool.pearsonr(res_zy,res_zx)[0]
        
        if (np.round(sttool.pearsonr(df4[:,0],df4[:,1])[0],5)==np.round(parcor_xyz,5)): 
            parcor_xyz = np.nan
            
        z0 = (1/2)*mt.log((1+parcor_xyz)/(1-parcor_xyz))
        test_stat = mt.sqrt(num_obs-1-3)*z0
        p_val = sttool.norm.sf(abs(test_stat))*2 ##two-sided
        sub_dta=[featIndex,parcor_xyz, p_val]
        appended_data.append(sub_dta)
        
    result_set = np.array(appended_data)
    
    ## remove nan element in result_set
    final_set = result_set[~np.isnan(result_set).any(axis=1)]
    best_split_index = final_set[final_set[:,1]==np.min(final_set[:,1])][0][0]
    return int(best_split_index)
                            
def createTree(df_original, dataSet, max_num_create, leafType=corLeaf, ops=(0.01,10)):
    global num_create
    num_create = num_create + 1
    
    if num_create > max_num_create : return leafType(dataSet)
       
    df_original = df_original
    feat, val = chooseBestSplit(df_original, dataSet, leafType, ops) 
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = df_original.columns[feat] 
    retTree['spVal'] = val
    if df_original[df_original.columns[feat]].dtypes=='object':
        lSet, rSet = cate_binSplitDataSet(df_original,dataSet,feat,val)
    else:
        lSet, rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(df_original,lSet, max_num_create, leafType, ops)
    retTree['right']=createTree(df_original, rSet, max_num_create, leafType,ops)
    
    return retTree
    
## this one does not consdier the tolerance to stop split #
###### NOTE THAT tolS is a fixed value, how much do two subgroup are different in correlation
##### TolN is the minimum sample size in each node. Both are controled in ops.    
def chooseBestSplit(df_original, dataSet, leafType=corLeaf, ops=(0.01,10)):
    df_original = df_original; tolN = ops[1]; tolS=ops[0]
    m,n = np.shape(dataSet)
       
    ####### Type 2 #######
    S = leafType(dataSet)**2
    bestS = -np.inf; bestIndex = 0; bestValue=0
    featIndex = Find_split_var_by_partialCor(df_original, dataSet)
    if featIndex == None: return None, leafType(dataSet) 
    if df_original[df_original.columns[featIndex]].dtypes=='object':
       categories = list(set(dataSet[:,featIndex].flat))
       poss_comb = sorted_k_partitions(categories,2)
            
       for splitVal in poss_comb:
           splitVal1 = splitVal[0]
           mat0, mat1 = cate_binSplitDataSet(df_original,dataSet, featIndex, splitVal1)
           if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] <tolN):
               continue
           newS = max(abs(leafType(mat0)), abs(leafType(mat1)))
           if newS > bestS : #since maximize newS
              bestIndex = featIndex
              bestValue = splitVal1
              bestS = newS

    else:        
        for splitVal in set(dataSet[:,featIndex].flat): #give value of the feature
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] <tolN):
                continue
                #### type 3 #### split creiteria 
            newS = max(abs(leafType(mat0)), abs(leafType(mat1)))
            if newS > bestS: #since maximize newS
               bestIndex = featIndex
               bestValue = splitVal
               bestS = newS
    
    if (bestS-S) < tolS:
        return None, leafType(dataSet) 
        
    if df_original[df_original.columns[bestIndex]].dtypes=='object':
        mat0, mat1 = cate_binSplitDataSet(df_original,dataSet, bestIndex, bestValue)
    else:
        mat0, mat1=binSplitDataSet(dataSet, bestIndex, bestValue)
    
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] <tolN) :
        return None, leafType(dataSet)
        
    return bestIndex, bestValue
        
###########################################################################

##### example ######
############ when applying, carefully construct dataset ##########
### set data X and Y into two first columns ###
df=pd.read_csv("C:/Users/Doowon/Documents/Python_DT/windta_comp.csv")
num_create= 0
df1=df.dropna() # remove na missing records
df2 = df1.as_matrix()

model3_1 = createTree(df, df2, 6, corLeaf, ops=(0.01,10))

  

