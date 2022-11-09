import os
import subprocess
import json
import ast

import model_util as mu
import predictor
import data

def getJson(fn):
    comDir = '/home/yangheechan/codeGen_web/flask/c2tok'
    flaskDir = '/home/yangheechan/codeGen_web/flask' 

    os.chdir(comDir)

    cmd = './query_gen.py ' + fn
    stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, encoding='utf-8')


    # prefix, postfix, label-type, prefix-text, postifx-text
    json_dat = json.dumps(ast.literal_eval(stdout))
    dict_dat = json.loads(json_dat)

    os.chdir(flaskDir)

    return dict_dat



overall_title = 'webModel'
title = 'boringssl_'+overall_title+'_all'

prefix_pack, postfix_pack, attn_pack = mu.getModel(overall_title, title)


for k in range(1, 6):
    fn = 'test' + str(k) + '.c'

    json_data = getJson(fn)


    pred_results = predictor.predict(
        json_data['prefix'],
        json_data['postfix'],
        prefix_pack,
        postfix_pack,
        attn_pack
    )

    final_json = data.returnFinalJson(json_data, pred_results)
    final = data.idx2str(pred_results)



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

    with open('test'+ str(k) + '_json', 'w') as f:
        info = json.dumps(final_json[0])
        f.write(info)
    