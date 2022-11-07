#!/usr/bin/python3

import sys
import os
import json


def main (target):
	argv = ['(',  'clang', '-fsyntax-only', '-Xclang', '-dump-tokens' ] 
	argv = argv + [target, '2>', '_tokens', ')']
	cmd = '' 
	for a in argv:
		cmd = cmd + ' ' + a
	os.system(cmd)
	os.system("grep \"" + target + "\" _tokens > " + "tokens")

if __name__ == "__main__":
	main(sys.argv[1])
