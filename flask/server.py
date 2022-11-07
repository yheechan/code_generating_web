from flask import Flask, jsonify, request

app = Flask(__name__)

from datetime import datetime
import os
import subprocess
import json

import model_util as mu
import predictor
import data


overall_title = 'webModel'
title = 'boringssl_'+overall_title+'_1'

prefix_pack, postfix_pack, attn_pack = mu.getModel(overall_title, title)


def makeFile(source_code):
    now = datetime.now()
    fn = now.strftime("%H_%M_%S") + '.c'

    with open('c2tok/'+fn, 'w') as f:
        f.write(source_code)
    
    return fn


def getInputs(fn):
    comDir = '/home/yangheechan/codeGen_web/flask/c2tok'
    flaskDir = '/home/yangheechan/codeGen_web/flask' 

    os.chdir(comDir)

    cmd = './query_gen.py ' + fn
    stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, encoding='utf-8')

    json_data = json.loads(stdout)
    prefix = json_data['prefix']
    postfix = json_data['postfix']

    os.chdir(flaskDir)

    return prefix, postfix


@app.route('/server/translate', methods=['POST'])
def generate():

    webData = request.get_json()
    source_code = webData['text']

    fn = makeFile(source_code)
    
    prefix, postfix = getInputs(fn)

    pred_results = predictor.predict(
        prefix,
        postfix,
        prefix_pack,
        postfix_pack,
        attn_pack
    )

    final = data.idx2str(pred_results)

    total = []
    for i in range(len(final)):
        out_str = 'patch #'+str(i+1) + ':\n' + final[i] + '\n'
        total.append(out_str)
    
    result = '\n'.join(total)

    print(result)

    return result


if __name__ == "__main__":
    app.run(debug=True)
