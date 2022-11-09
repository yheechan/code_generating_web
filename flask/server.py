from flask import Flask, jsonify, request

app = Flask(__name__)

from datetime import datetime
import os
import subprocess
import json
import ast

import model_util as mu
import predictor
import data


overall_title = 'webModel2'
title = 'boringssl_'+overall_title+'_1'

prefix_pack, postfix_pack, attn_pack = mu.getModel(overall_title, title)


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

    webData = request.get_json()
    source_code = webData['text']

    fn = makeFile(source_code)
    
    json_data = getJson(fn)

    pred_results = predictor.predict(
        json_data['prefix'],
        json_data['postfix'],
        prefix_pack,
        postfix_pack,
        attn_pack
    )

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

    return result


if __name__ == "__main__":
    app.run(debug=True)
