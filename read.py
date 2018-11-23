# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:49:06 2018

@author: Sayan_Banerjee03
"""
#import pandas as pd
data_file = 'D:/Data Science/MS_AI_Challenge/data/data.tsv'
sample_data_file = 'D:/Data Science/MS_AI_Challenge/interim/sample_data.tsv'
trainFileName = "D:/Data Science/MS_AI_Challenge/interim/traindata.tsv"
validationFileName = "D:/Data Science/MS_AI_Challenge/interim/validationdata.tsv"

# =============================================================================
# def read_in_chunks(file_object, chunk_size=1024):
#     """Lazy function (generator) to read a file piece by piece.
#     Default chunk size: 1k."""
#     while True:
#         data = file_object.read(chunk_size)
#         if not data:
#             break
#         yield data
# 
# 
# f = open(data_file,encoding="latin-1")
# for piece in read_in_chunks(f):
#     print(piece)
#     break
# =============================================================================
sample_list = list()
i = 0
with open(data_file,encoding="utf-8") as f:
    for line in f:
        sample_list.append(line)
        i += 1
        if (i >= 10000):
            break

with open(sample_data_file, 'w', encoding="utf-8") as f:
    for item in sample_list:
        f.write("%s" % item)

# =============================================================================
# data =   pd.read_csv(data_file,sep = '\t')
# sample_df = data.iloc[0:5]
# =============================================================================
train_list= list()
test_list = list()
old_q_id = -1
q_no = 0
with open(data_file,encoding="utf-8") as f:
    for line in f:
        q_id = line.strip().lower().split("\t")[0]
        if (old_q_id != q_id):
            q_no += 1
            old_q_id = q_id
        if (q_no % 10 == 0):
            test_list.append(line)
        else:
            train_list.append(line)
print("total number of queries:", q_no)

with open(trainFileName, 'w', encoding="utf-8") as f:
    for item in train_list:
        f.write("%s" % item)
        
with open(validationFileName, 'w', encoding="utf-8") as f:
    for item in test_list:
        f.write("%s" % item)
