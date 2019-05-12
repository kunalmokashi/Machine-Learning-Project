import pandas as pd
import numpy as np
from numpy import NaN

def check_reg(unique_list):
    for item in unique_list:
        try:
            float(item)
        except ValueError:
            return False
    return True

def conv_col(unique_list):
    __dict_null={' ':-1,'.':-1,'prefernot':-1}

    for i in __dict_null.keys():
        unique_list=unique_list.replace(i,__dict_null.get(i))

    _temp_list=unique_list.unique()
    _dict={}
    
    if len(_temp_list)>40 and check_reg(_temp_list):
        _temp_list=pd.to_numeric(_temp_list,errors="coerce")
        unique_list=pd.to_numeric(unique_list,errors="coerce")
        mx=max(_temp_list)
        mn=min(_temp_list)
        diff=mx-mn
        bins=10
        step=diff/bins
        ranges = ["[{0} - {1})".format(idx,  idx+step) for idx in np.arange(mn, mx, step)]
        _temp_unique_list=pd.cut(unique_list,bins,labels=ranges)
        val=0
        for key in _temp_unique_list.unique():
            if (key!=-1):
                _dict[key]=val+1
                val+=1
            else:
                _dict[key]=NaN
        unique_list=_temp_unique_list
    else:
        val=0
        for key in _temp_list:
            if (key!=-1):
                _dict[key]=val+1
                val+=1
            else:
                _dict[key]=NaN

    for i in _dict.keys():
        unique_list=unique_list.replace(i,_dict.get(i))

    unique_list=pd.to_numeric(unique_list,errors="coerce")
    if -1 in list(_dict.keys()):
        _dict.pop(-1)

    return unique_list,_dict

def preprocess_data(path):
    temp_data=pd.read_csv(path, low_memory=False)
    with open("column_list.txt",'r') as f:
        temp_list=f.read()
        column_list=list(temp_list.strip("\'][\'").split("\', \'"))
    
    data=temp_data[column_list]
    codebook=pd.DataFrame(index=column_list,columns=['Codeword_Dict'])
    
    predata=pd.DataFrame(columns=column_list)
    
    for col in column_list:
        predata[col],codebook.at[col,'Codeword_Dict']=conv_col(data[col])
#    print(codebook)
    codebook.to_csv("ML1/ML1/CodebookData.csv")
    predata.to_csv("ML1/ML1/ProcessedData.csv")
    return predata, codebook