import json
import numpy as np
from collections import deque
import csv

def insertSOSandEOS(tok_list):
    tmp = deque(tok_list)
    tmp.appendleft(213)
    tmp.append(214)
    tmp = list(tmp)
    return tmp

def insertEOS(tok_list, idx):
    tmp = tok_list
    tmp.insert(idx, 1)
    return tmp

def getTrainData(proj_list, target_project):

    total_file = 'total'

    prefix = []
    postfix = []
    label_type = []
    label_len = []

    for proj in proj_list:
        
        # if proj == target_project or proj == total_file: continue

        # don't remove target file for training web model
        if proj == total_file: continue

        print('Getting data for \"' + target_project + '\" from \"' + proj + '\"')

        with open('../data/' + proj, 'r') as f:
            lines = f.readlines()
        
        for line in lines:

            json_data = json.loads(line.rstrip())

            # prefix.append(insertSOSandEOS(json_data['prefix']))
            # postfix.append(insertSOSandEOS(json_data['postfix']))

            prefix.append(json_data['prefix'])

            postfix_data = json_data['postfix']
            postfix_data.pop()
            postfix_data.insert(0, 0)
            postfix.append(postfix_data)

            label_type.append(json_data['label-type'])

            label_len.append(json_data['label-len'])
    
        # ------------------------------------------------------
        # break for reducing test time for quick development
        # break
    
    return np.array(prefix), np.array(postfix), np.array(label_type), np.array(label_len)

def getTestData(target_project):
    prefix = []
    postfix = []
    label_type = []
    label_len = []

    with open('../data/' + target_project, 'r') as f:

        lines = f.readlines()
    
    for line in lines:

        json_data = json.loads(line.rstrip())

        # prefix.append(insertSOSandEOS(json_data['prefix']))
        # postfix.append(insertSOSandEOS(json_data['postfix']))

        prefix.append(json_data['prefix'])

        postfix_data = json_data['postfix']
        postfix_data.pop()
        postfix_data.insert(0, 0)
        postfix.append(postfix_data)

        label_type.append(json_data['label-type'])

        label_len.append(json_data['label-len'])
    
    return np.array(prefix), np.array(postfix), np.array(label_type), np.array(label_len)

def getInfo():

    max_len = 0
    source_code_tokens = []
    token_choices = []

    with open('../record/max_len', 'r') as f:
        max_len = int(f.readline().rstrip())
    
    with open('../record/source_code_tokens', 'r') as f:
        source_code_tokens = [int(line.rstrip()) for line in f]
    
    with open('../record/token_choices', 'r') as f:
        token_choices = [int(line.rstrip()) for line in f]

    return max_len, source_code_tokens, token_choices

def getIdx2str():
    idx2str = {}

    with open('../record/token_str', 'r') as f:
        csvReader = csv.reader(f)

        for row in csvReader:
            # if row[1] != '':
            idx2str[int(row[0])] = row[1]
    
    return idx2str



def idx2str(label_results, total_colored):
    idx2str = getIdx2str()

    final = []

    for i in range(len(label_results)):
        str_list = []
        seq = label_results[i]

        for j in range(len(seq)):
            token_num = seq[j]

            if token_num != 2:
                str_list.append(idx2str[token_num])
            elif token_num == 2:
                str_list.append(total_colored[i][j])

        str = ' '.join(str_list) 
        final.append(str)
    
    return final

def returnFinalJson(json_data, label_results):
    finalJson = []


    for seq in label_results:
        json_data['label-type'] = seq + [0]*(10-len(seq))
        json_data['label-length'] = len(seq)
        finalJson.append( json_data.copy() )
    
    return finalJson

def strip(label_results):
    tot_stripped = []

    for i in range(len(label_results)):
        b4zero = []
        seq = label_results[i]

        for j in range(len(seq)):
            token_num = seq[j]

            if token_num == 0:
                break

            b4zero.append(token_num)

        tot_stripped.append(b4zero) 
    
    return tot_stripped