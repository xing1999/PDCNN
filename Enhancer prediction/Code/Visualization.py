import torch
import numpy as np
import torch.nn as nn
import random
from model import MyModel
from Dataset import load_data,tran_POCD
import math
import umap
import matplotlib.pyplot as plt

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






test_sequence_list,test_all_matrix_input,test_label_list=load_data(".\\human\\200bp\\test_neg.tsv",".\\human\\200bp\\test_pos.tsv")
# test_sequence_list,test_all_matrix_input,test_label_list=load_data(".\\2L_data\\two\\test_neg.tsv",".\\2L_data\\two\\test_pos.tsv")
test_list_info = ObtainRandom(len(test_label_list))

batch_size = len(test_sequence_list)
device = torch.device('cuda')
torch.manual_seed(1234)
model = MyModel().to(device)
model.load_state_dict(torch.load("1.pt"))
model.eval()

criteon = nn.CrossEntropyLoss().to(device)

with torch.no_grad():

    TP, FN, FP, TN = 0, 0, 0, 0
    batch_num = math.ceil(len(test_list_info) / batch_size)

    for idx in range(batch_num):


        x = Catbatch(test_all_matrix_input, test_list_info, idx, batch_size, batch_num)
        label = Catbatch(test_label_list, test_list_info, idx, batch_size, batch_num)
        x, label = x.to(device), label.to(device)

        out1, out2, logits = model(x)
        pred = logits.argmax(dim=1)

        x_original = x.clone().detach().cpu().numpy().reshape(x.size(0), -1)
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        embedding_original = umap_model.fit_transform(x_original)
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_original[:, 0], embedding_original[:, 1], c=label.cpu().numpy(), cmap='viridis', s=10)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('PDCNN Visualization of Original input')
        plt.savefig('PDCNN_visualization_Original_input.png')
        plt.show()


        # conv1_output = model.conv1(x)
        conv1_output_np = out1.cpu().numpy()
        flattened_conv1_output = conv1_output_np.reshape(conv1_output_np.shape[0], -1)
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        embedding_conv1 = umap_model.fit_transform(flattened_conv1_output)
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_conv1[:, 0], embedding_conv1[:, 1], c=label.cpu().numpy(), cmap='viridis', s=10)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('PDCNN Visualization of Conv1 Output')
        plt.savefig('PDCNN_visualization_Conv1_Output.png')
        plt.show()

        conv1_output_np = out2.cpu().numpy()
        flattened_conv1_output = conv1_output_np.reshape(conv1_output_np.shape[0], -1)
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        embedding_conv1 = umap_model.fit_transform(flattened_conv1_output)
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_conv1[:, 0], embedding_conv1[:, 1], c=label.cpu().numpy(), cmap='viridis', s=10)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('PDCNN Visualization of Conv2 Output')
        plt.savefig('PDCNN_visualization_Conv2_Output.png''C2.png')
        plt.show()


        fc_output_np = logits.cpu().numpy()
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        embedding_fc = umap_model.fit_transform(fc_output_np)

        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_fc[:, 0], embedding_fc[:, 1], c=label.cpu().numpy(), cmap='viridis', s=10)
        # plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        # title_font = {'fontname': 'Airl', 'size': '16', 'color': 'black'}
        plt.title('PDCNN Visualization of FC Output')
        plt.savefig('PDCNN_Visualization_of_FC_Output.png')
        plt.show()

