from tkinter.ttk import Label
import torch
import torch.nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import sys
import argparse

import numpy as np
import math
import random
import json

import models
import models.seq2seq_point_bce

PROJECT_NAME= 'mutation'

class SingleDatapoint(Dataset):
    def __init__(self, datapoint):
        self.length = 64
        self.data = [ datapoint ]
        #print(self.data)

    def __getitem__(self,idx):
        pre, post, label_type, label_prefix, label_postfix, case = self.data[idx]
        #print(pre, post, label_type, label_prefix, label_postfix, case)
        # return torch.LongTensor(pre), torch.LongTensor(post), torch.FloatTensor(label_prefix), torch.FloatTensor(label_postfix)
        return (
            torch.LongTensor(pre[-1*self.length:]),
            torch.LongTensor(post[-1*self.length:]),
            torch.LongTensor(label_type),
            torch.LongTensor(label_prefix[-1*self.length:]),
            torch.LongTensor(label_postfix[-1*self.length:]),
                )

    def __len__(self):
        return len(self.data)

def read_query_file(querypath):
    with open(querypath, 'r') as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            return line 
            #prefix = line['prefix']
            #postfix = line['postfix']
            #label_type = line['label-type']
            #label_prefix = line['label-prefix']
            #label_postfix = line['label-postfix']
            #case = line['case']
            #return [prefix, postfix, label_type, label_prefix, label_postfix, case]
     
def eval_pointer(model, dataset):
    model.eval()

    for i, (pre, post, label_type, label_pre, label_post) in enumerate(dataset, 1):
        with torch.no_grad():
            pre_out, post_out = model(pre, post, label_type, teacher_forcing_ratio=0)
            break
    pre_out = pre_out.tolist()[0][0]
    post_out = post_out.tolist()[0][0]

    pre_out = (lambda x: [i[0] for i in x])(pre_out)
    post_out = (lambda x: [i[0] for i in x])(post_out)

    return {'pre' : pre_out, 'post': post_out}

def get_random_score(pointer_query):
    pre = []
    post = []
    for i in range(0, 64):
        pre.append(random.random())
        post.append(random.random())
    return [pre, post]    

def get_pointer_score(pointer_query):
    model = models.Seq2SeqPointAttnBCE(vocab_size=213, embedding_dim=256, hidden_dim=512)
    models.init_weights(model)
    #model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('best.pt'))

    pointer_query[0] = pointer_query[0][-64:]
    pointer_query[1] = pointer_query[1][-64:]

    testset = SingleDatapoint(pointer_query) 

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, drop_last = False)
    r = eval_pointer(model, testloader)
    return r 

def get_fittest_text(pq, pq_text, r):
    prefix_min_idx = None
    prefix_min_val = None
    for i in range(0, 64):
        if pq[0][i] == 2 and pq_text[0][i] != '':
            if prefix_min_idx == None or r[0][i] < prefix_min_val:
                prefix_min_idx = i
                prefix_min_val = r[0][i]
    
    postfix_min_idx = None
    postfix_min_val = None
    for i in range(0, 64):
        if pq[1][i] == 2 and pq_text[1][i] != '':
            if postfix_min_idx == None or r[1][i] < postfix_min_val:
                postfix_min_idx = i
                postfix_min_val = r[1][i]
   
    if prefix_min_val == None and postfix_min_val == None:
        return '$$'
    elif prefix_min_val != None and postfix_min_val == None:
        return pq_text[0][prefix_min_idx]
    elif prefix_min_val == None and postfix_min_val != None:
        return pq_text[1][postfix_min_idx] 
    elif prefix_min_val < postfix_min_val:
        return pq_text[0][prefix_min_idx]
    else:
        return pq_text[1][postfix_min_idx]
    return '$$'

#######

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]=""

    q = read_query_file('query')

    label_seq = q['label-type'][0:q['label-length']]
    label_seq.reverse()
    pq = [q['prefix'], q['postfix'] + label_seq, [], [0] * 64, [0] * 64, 2]
    pq_text = [q['prefix-text'], q['postfix-text'] + [''] * q['label-length']]

    label_text = [''] * 10
    
    #for i in range(0, 1):
    for i in range(0, int(q['label-length'])):
        if q['label-type'][i] == 2:
            pq[2] = pq[1][-1] 
            pq[1].pop()
            pq_text[1].pop()
            #r = get_pointer_score(pq)
            r = get_random_score(pq)
            text = get_fittest_text(pq, pq_text, r)
            label_text[i] = text
            pq[0].append(pq[2])
            pq_text[0].append(text)
            #q['label-text'].append = text
        else:
            pq[1].pop()
            pq[0].append('')
            #q['label-text'].append('')
            label_text[i] = '' 

    #ps = get_pointer_score(pointer_query) 
    print(label_text)
