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
lin_num = 0
with open(data_file,encoding="utf-8") as f:
    for line in f:
        q_id = int(line.strip().lower().split("\t")[0].strip())
        lin_num += 1
        if q_id == 406184:
            print(lin_num)

with open(sample_data_file, 'w', encoding="utf-8") as f:
    for item in sample_list:
        f.write("%s" % item)

# =============================================================================
# data =   pd.read_csv(data_file,sep = '\t')
# sample_df = data.iloc[0:5]
# =============================================================================

######## This part of code is to create training an validation set.
train_list= list()
test_list = list()
old_q_id = -1
q_no = 0
ans_num = 0
is_test = False
is_train = False
train_dict_q = {}
test_dict_q = {}
with open(data_file,encoding="utf-8") as f:
    for line in f:
        q_id = int(line.strip().lower().split("\t")[0].strip())
        ans_num += 1
        if (old_q_id != q_id) and (q_id not in train_dict_q) and (q_id not in test_dict_q):
            if ans_num < 10:
                print("old question id: %d new question id: %d and number of answers: %d"\
                      %(old_q_id,q_id,ans_num))
                if is_train:
                    train_dict_q[old_q_id] = train_dict_q.get(old_q_id, 0) + ans_num
                if is_test:
                    test_dict_q[old_q_id] = test_dict_q.get(old_q_id, 0) + ans_num
            q_no += 1
            old_q_id = q_id
            ans_num = 0
        if (q_id in test_dict_q) or ((q_id not in train_dict_q) and (q_no % 10 == 0)):
            test_list.append(line)
            is_test = True
            is_train = False
            if q_id in test_dict_q:
                test_dict_q[q_id] +=1
        else:
            train_list.append(line)
            is_test = False
            is_train = True
            if q_id in train_dict_q:
                train_dict_q[q_id] +=1
print("total number of queries:", q_no)

with open(trainFileName, 'w', encoding="utf-8") as f:
    for item in train_list:
        f.write("%s" % item)
        
with open(validationFileName, 'w', encoding="utf-8") as f:
    for item in test_list:
        f.write("%s" % item)
