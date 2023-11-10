import torch
import numpy as np
import torch.nn as nn
import random
from model import MyModel
from Dataset import load_data,tran_POCD
import math
import csv

def ObtainRandom(length):
    list_info = []
    while True:
        info = random.randint(0,length-1)
        if info not in list_info:
            list_info.append(info)
        if len(list_info) ==length:
            break

    return list_info

def Catbatch(input_list, info_list, idx, batch_size, batch_num):


    if idx ==  (batch_num - 1):
        batch = info_list[idx * batch_size: ]
    else:
        batch = info_list[idx * batch_size: (idx + 1) * batch_size]

    for idx, x in enumerate(batch):

        if idx == 0:
            catbatch = torch.unsqueeze(input_list[x], dim=0)

        else:
            cattensor = torch.unsqueeze(input_list[x], dim=0)
            catbatch = torch.cat((catbatch, cattensor), dim = 0)


    return catbatch

def eff(labels, preds):

    TP, FN, FP, TN = 0, 0, 0, 0

    for idx,label in enumerate(labels):

        if label == 1:
            if label == preds[idx]:
                TP += 1
            else: FN += 1
        elif label == preds[idx]:
            TN += 1
        else: FP += 1

    return TP, FN, FP, TN


def Judeff(TP, FN, FP, TN):

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    MCC = (TP * TN - FP * FN) / (math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))

    return SN, SP, ACC, MCC

def Calauc(labels, preds):

    labels = labels.clone().detach().cpu().numpy()
    preds = preds.clone().detach().cpu().numpy()

    f = list(zip(preds, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    pos_cnt = np.sum(labels == 1)
    neg_cnt = np.sum(labels == 0)
    AUC = (np.sum(rankList) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)

    return AUC


test_sequence_list,test_all_matrix_input,test_label_list=load_data(".\\human\\200bp\\test_neg.tsv",".\\human\\200bp\\test_pos.tsv")
# test_sequence_list,test_all_matrix_input,test_label_list=load_data(".\\2L_data\\two\\test_neg.tsv",".\\2L_data\\two\\test_pos.tsv")
test_list_info = ObtainRandom(len(test_label_list))

batch_size = len(test_sequence_list)
device = torch.device('cuda')
torch.manual_seed(1234)
model = MyModel().to(device)
model.load_state_dict(torch.load("H_200_4.pt"))
model.eval()

criteon = nn.CrossEntropyLoss().to(device)

with torch.no_grad():

    TP, FN, FP, TN = 0, 0, 0, 0
    batch_num = math.ceil(len(test_list_info) / batch_size)

    for idx in range(batch_num):
         # x = Catbatch(test_One_hot_matrix_input, test_list_info, idx, batch_size, batch_num)
        # x = Catbatch(test_NCP_matrix_input, test_list_info, idx, batch_size, batch_num)
        # x = Catbatch(test_DPCP_matrix_input, test_list_info, idx, batch_size, batch_num)
        x = Catbatch(test_all_matrix_input, test_list_info, idx, batch_size, batch_num)
        label = Catbatch(test_label_list, test_list_info, idx, batch_size, batch_num)
        x, label = x.to(device), label.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        A, B, C, D = eff(label, pred)
        TP += A
        FN += B
        FP += C
        TN += D
        AUC = Calauc(label, pred)

    SN, SP, ACC, MCC = Judeff(TP, FN, FP, TN)

    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))

    print("SN: {}, SP: {}, ACC: {}, MCC: {}, AUC: {}".format(SN, SP, ACC, MCC, AUC))
# modelname='NCP-H'
# date=[modelname,TP,FN,FP,TN,SN, SP, ACC, MCC, AUC]
# csvfile = open('rundate.csv', 'a', encoding='utf-8',newline='')
# writer = csv.writer(csvfile)
# # writer.writerow(['Model','TP', 'FN', 'FP', 'TN','SN', 'SP','ACC','MCC','AUC'])
# writer.writerow(date)
# csvfile.close()

