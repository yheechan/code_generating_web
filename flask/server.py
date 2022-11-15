from flask import Flask, jsonify, request

app = Flask(__name__)

from datetime import datetime
import os
import subprocess
import json
import ast
import torch

import model_util as mu
import predictor
import data


overall_title = 'reModelSeq2Seq_tune'
title = 'boringssl_'+overall_title+'_tryOVF'

model = mu.getModel(overall_title, title)


device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def makeFile(source_code):
    now = datetime.now()
    fn = now.strftime("%H_%M_%S") + '.c'

    with open('c2tok/'+fn, 'w') as f:
        f.write(source_code)
    
    return fn


def getJson(fn):
    comDir = '/home/yangheechan/codeGen_web/flask/c2tok'
    flaskDir = '/home/yangheechan/codeGen_web/flask' 

    os.chdir(comDir)

    cmd = './query_gen.py ' + fn
    stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, encoding='utf-8')

    # prefix, postfix, label-type, prefix-text, postifx-text
    json_data = json.dumps(ast.literal_eval(stdout))
    dict_data = json.loads(json_data)

    os.chdir(flaskDir)

    return dict_data


def writeJson(json_data):
    with open('identifier_gen/query', 'w') as f:
        info = json.dumps(json_data)
        f.write(info)


def getidentifier2Str(patch_list):
    genDir = '/home/yangheechan/codeGen_web/flask/identifier_gen'
    flaskDir = '/home/yangheechan/codeGen_web/flask' 

    label_type = 'label-type'

    total_str = []

    for patch_dict in patch_list:
        
        writeJson(patch_dict)

        os.chdir(genDir)

        cmd = 'python3 identifier_gen.py'
        stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, encoding='utf-8')

        # prefix, postfix, label-type, prefix-text, postifx-text
        str_colored = json.dumps(ast.literal_eval(stdout))
        list_colored = ast.literal_eval(str_colored)

        os.chdir(flaskDir)

        total_str.append(list_colored)

    return total_str


@app.route('/server/translate', methods=['POST'])
def generate():
    print('start generate expression')

    webData = request.get_json()
    source_code = webData['text']

    fn = makeFile(source_code)
        
    json_data = getJson(fn)


    pred_results = predictor.myBeamStart(
        model,
        json_data['prefix'],
        json_data['postfix'],
        device=device,
        beam_width=5
    )

    pred_results = data.strip(pred_results)
    final_json = data.returnFinalJson(json_data, pred_results)
    total_colored = getidentifier2Str(final_json)
    final = data.idx2str(pred_results, total_colored)

    total = []
    for i in range(len(final)):
        out_str = 'patch #'+str(i+1) + ':\n' + final[i] + '\n'
        total.append(out_str)
    
    result = '\n'.join(total)

    for i in range(len(final_json)):
        print('patch (label-type) #' + str(i+1) + ':')
        print(final_json[i]['label-type'])
        print()

    print(result)

    print('-------------------------------------\n')

    return {
        "patch_1": final[0],
        "patch_2": final[1],
        "patch_3": final[2],
        "patch_4": final[3],
        "patch_5": final[4]
    }


if __name__ == "__main__":
    app.run(debug=True)
