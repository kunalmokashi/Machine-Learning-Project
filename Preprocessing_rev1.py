import pandas as pd
import numpy as np
from math import nan

def check_reg(unique_list):
    for item in unique_list:
        try:
            int(item)
        except ValueError:
            return False
    return True

def conv_col(unique_list):
    __dict_null={' ':None,'.':None,'prefernot':None}
    # print(unique_list.dtype)

    for i in __dict_null.keys():
        unique_list=unique_list.replace(i,__dict_null.get(i))

    _temp_list=unique_list.unique()
    _dict={}
    
    if len(_temp_list)>40:
        if check_reg(_temp_list):
            _temp_list=pd.to_numeric(_temp_list,errors="coerce")
            unique_list=pd.to_numeric(unique_list,errors="coerce")
            # print(isinstance(_temp_list,np.ndarray))
            mx=max(_temp_list)
            mn=min(_temp_list)
            diff=mx-mn
            bins=10
            step=np.ceil(diff/bins)
            ranges = ["[{0} - {1})".format(idx,  idx+step) for idx in np.arange(mn, mx-5, step)]
            print(ranges,len(ranges))
            _temp_unique_list=pd.cut(unique_list,bins,labels=ranges)
            # print(_temp_unique_list,_temp_unique_list.unique())
            for val,key in enumerate(_temp_unique_list.unique()):
                if (key != None):
                    _dict[key]=val+1
                else:
                    _dict[key]=nan
            print(_dict)
            unique_list=_temp_unique_list
    # temp_list.sort()
    else:
        for val,key in enumerate(_temp_list):
            if (key != None):
                _dict[key]=val+1
            else:
                _dict[key]=nan
    # print(_dict)

    for i in _dict.keys():
        unique_list=unique_list.replace(i,_dict.get(i))

    unique_list=pd.to_numeric(unique_list,errors="coerce")
    # print(unique_list.dtype,unique_list)    

    return unique_list,_dict



temp_data=pd.read_csv("..//FINALE//CleanedDataset_v2_processed.csv",low_memory=False)
# print(data)
# temp_list=data.columns
# col_list=[temp_list[i] for i in range(len(temp_list))]
# print(col_list)
with open("..//FINALE//column_list.txt",'r') as f:
    # f.write(str(col_list))
    temp_list=f.read()
    column_list=list(temp_list.strip("\'][\'").split("\', \'"))

data=temp_data[column_list]
# print(data)
# row_idx=data.columns
codebook=pd.DataFrame(index=column_list,columns=['Codeword_Dict'])
# print(codebook)
# print(col_list)

predata=pd.DataFrame(columns=column_list)

for col in column_list:
    # print(col)
    # col=str(col)
    # print(data[col])
    predata[col],codebook.at[col,'Codeword_Dict']=conv_col(data[col])

    # print(codebook.at[col,'Codeword_Dict'])

print(codebook)
codebook.to_csv("..//FINALE//CodebookData.csv")
predata.to_csv("..//FINALE//ProcessedData.csv")







# codebook.to_csv("..//FINALE//CodebookData.csv",index=False)
