import torch
import numpy as np
import random


def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))
    return sequences, labels

def make_sequences_equal_length(sequences, N):
    max_length = max(len(sequence) for sequence in sequences)

    new_sequences = []
    for sequence in sequences:
        if len(sequence) < max_length:
            padding = N * (max_length - len(sequence))
            new_sequence = sequence + padding
        else:
            new_sequence = sequence
        new_sequences.append(new_sequence)
    return new_sequences
def generate_kmers(sequence, k):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        kmers.append(kmer)
    # print(kmers)
    return kmers


def generate_vocab(sequences, k):
    vocab = set()
    for sequence in sequences:
        kmers = generate_kmers(sequence, k)
        vocab.update(kmers)
    vocab = [item for item in vocab if 'N' not in item]
    return sorted(list(vocab))

def count_total_kmers(sequences, k):
    total_kmers = 0
    for sequence in sequences:
        total_kmers += len(sequence) - k + 1
    return total_kmers



def generate_frequency_matrix(vocab, sequences):
    positions = max(len(sequence) for sequence in sequences)
    # print("positions:{}".format(positions))
    matrix = [[0] * len(vocab) for _ in range((positions-k+1))]
    for i, K_mer in enumerate(sequences):
        kmers=generate_kmers(K_mer, k)
        for j, k_mer in enumerate(kmers):
            for p,Vocab in enumerate(vocab):
                if Vocab == k_mer:
                    matrix[j][p]=matrix[j][p]+1

    return matrix


def Generate_posden(vocab,sequences,k):
    NS = count_total_kmers(sequences, k)
    Z=generate_frequency_matrix(vocab,sequences)
    Z_den = []
    for row in Z:
        new_row = [element / NS for element in row]
        Z_den.append(new_row)
    # print("vocab:{}".format(vocab))
    # print("NS:{}".format(NS))
    # for row in Z:
    #     print(row)
    return Z_den
def calculate_POCD(K1, K2):
    rows = len(K1)
    cols = len(K1[0])
    p = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            min_val = min(K1[i][j], K2[i][j])
            if min_val == 0:
                min_val = 0.1
            p[i][j] = (K1[i][j] - K2[i][j]) / min_val

    return p
def get_POCD(Vocab,pos_seq,neg_seq,k):
    Z_posden = Generate_posden(Vocab, pos_seq, k)
    Z_negden = Generate_posden(Vocab, neg_seq, k)
    p = calculate_POCD(Z_posden, Z_negden)
    return p

def encode(sequences,vocab,k,p):
    PoCD = []
    for seq in sequences:
        # print(seq)
        l=len(seq)
        k_mers=generate_kmers(seq, k)
        # print(k_mers)
        Pocd = np.zeros([l-k+1])
        for x, k_mer in enumerate(k_mers):
            for w,voc in enumerate(vocab):
                if k_mer==voc:
                    Pocd[x]=p[x][w]
        PoCD.append(Pocd)
        # print(Pocd)
    return PoCD
def get_seq(neg_path,pos_path):
    old_sequences = []
    label = []
    # old_pos_seq=["TCGTA","ACGTA"]
    # old_neg_seq=["TTACT","TTCGT"]

    old_neg_seq, l1 = load_tsv_format_data(neg_path)
    old_pos_seq, l2 = load_tsv_format_data(pos_path)
    old_sequences.extend(old_pos_seq)
    old_sequences.extend(old_neg_seq)
    label.extend(l2)
    label.extend(l1)

    # 将序列补齐至等长
    sequences = make_sequences_equal_length(old_sequences, N)
    pos_seq = make_sequences_equal_length(old_pos_seq, N)
    neg_seq = make_sequences_equal_length(old_neg_seq, N)
    return old_sequences,sequences,pos_seq,neg_seq,label
# Print vocabulary
# Example data
def tran_POCD(neg_path, pos_path):
    old_sequences, sequences, pos_seq, neg_seq, label = get_seq(neg_path, pos_path)
    Vocab = generate_vocab(sequences, k)
    POCD = get_POCD(Vocab, pos_seq, neg_seq, k)
    return POCD

def load_data(neg_path,pos_path):#,POCD):

    old_sequences, sequences, pos_seq, neg_seq, label = get_seq(neg_path, pos_path)
    Vocab = generate_vocab(sequences, k)
    POCD=get_POCD(Vocab,pos_seq,neg_seq,k)
    data_encode=encode(old_sequences,Vocab,k,POCD)

    data_input = [torch.from_numpy(array) for array in data_encode]
    label_list_input = np.asarray([i for i in label], dtype=np.int64)
    label_list_input = torch.from_numpy(label_list_input)


    # print(data_input)
    # print(sequences)
    # print(label_list_input)
    # cols = len(data_encode[0])
    # print("{}列".format(cols))
    return sequences,data_input,label_list_input

k = 4
N = 'N'

