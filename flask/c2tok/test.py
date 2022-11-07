import os
import subprocess
import json

def getInputs(fn):
    comDir = '/home/yangheechan/codeGen_web/flask/c2tok'
    flaskDir = '/home/yangheechan/codeGen_web/flask' 

    os.chdir(comDir)

    cmd = './query_gen.py ' + fn

    # result = os.system(cmd)
    std_o = subprocess.check_output(cmd)
    print(std_o)


    data = json.loads(std_o)
    prefix = data['prefix']
    postfix = data['postfix']

    print(prefix)
    print(postfix)

    os.chdir(flaskDir)


getInputs('test.c')