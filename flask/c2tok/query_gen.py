#!/usr/bin/python3 

import sys
import re
import csv
import os

######
token_list = list()
query_list = list()
token_dict = dict() 
dp_token_len = 64

#label_len = 8
label_len = 10


identifier_types = \
	['identifier', 'raw_identifier', 'char_constant', \
	'wide_char_constant', 'utf8_char_constant', 'utf16_char_constant', \
	'utf32_char_constant', 'string_literal', 'wide_string_literal', \
	'header_name', 'utf8_string_literal', 'utf16_string_literal', \
	'utf32_string_literal'] 

#####
def load_dictionary (fdict):
	global token_dict

	with open(fdict) as f:
		c = csv.reader(f, delimiter='\t')
		for r in c:
			token_dict[r[0].rstrip()] = int(r[1])

def line_to_token (s):
	t = dict() 

	s1 = s.split(' ', 1)
	t['type'] = s1[0]

	lword = s1[1].find('\'') + 1 
	rword = s1[1].rfind('\'')
	t['word'] = s1[1][lword : rword]

	lword = s1[1].rfind('Loc=<') + 5

	pos_tokens = re.split(r':|<', s1[1][lword : -2]) 
	try:
		t['position'] = dict()
		t['position']['file'] = pos_tokens[0]
		t['position']['line'] = int(pos_tokens[1]) 
		t['position']['offset'] = int(pos_tokens[2].strip('>'))
	except IndexError:
		print("IndexError\n", file=sys.stderr)
		print(s + "\n", file=sys.stderr)
		sys.exit(1)
	
	#t['code'] = 0 

	return t


def line_to_query (s):
	tokens = re.split(',|:|\n', s) 

	query = dict()
	query['begin'] = dict()
	query['end'] = dict()
	query['file'] = tokens[0] 

	query['begin']['line'] = int(tokens[1])
	query['begin']['offset'] = int(tokens[2])
	query['end']['line'] = int(tokens[4])
	query['end']['offset'] = int(tokens[5])

	return query 


def token_to_query (t):
	query = dict() 

	query['begin'] = dict()
	query['end'] = dict()
	query['file'] = t['position']['file']

	query['begin']['line'] = t['position']['line']
	query['begin']['offset'] = t['position']['offset']
	query['begin']['file'] = t['position']['file'] 
	query['end']['line'] = t['position']['line'] 
	query['end']['offset'] = t['position']['offset']
	query['end']['file'] = t['position']['file'] 

	return query 


def read_tokens (ftoken):
	global token_list 

	fp = open(ftoken, 'r')
	for l in fp:
		token = line_to_token(l)
		token_list.append(token)
		#print(token, end='\n')


def read_quries ():
	global token_list, query_list

	for t in token_list:
		if t['type'] == 'identifier' and t['word'] == '$$':
			query_list.append(token_to_query(t))


def less_than (loc1, loc2):
	if loc1['file'] != loc2['file']:
		return True
	if loc1['line'] < loc2['line']:
		return True 
	elif loc1['line'] > loc2['line']:
		return False
	if loc1['offset'] < loc2['offset']:
		return True
	return False 


def convert_token (t):
	global token_dict

	if t['type'] in identifier_types:
		#if t['word'] in token_dict:
		#	return token_dict[t['word']]
		return token_dict['@unk']

	if not(t['type'] in token_dict):
		return token_dict['@ext']
		
	return token_dict[t['type']] 


def check_identity (t1, t2):
	if t1['type'] == t2['type']:
		if t1['word'] == t2['word']:
			return 1
	return 0 


def generate_datapoint (query):
	global token_list, dp_token_len

	dp = dict() 
	dp['prefix'] = list() 
	dp['prefix'] = [ token_dict['@pad'] ] * dp_token_len 
	dp['postfix'] = list()
	dp['postfix'] = [ token_dict['@pad'] ] * dp_token_len
	dp['label-type'] = list()
	dp['label-type'] = [ token_dict['@pad'] ] * label_len

	#dp['label-prefix'] = list()
	#dp['label-postfix'] = list()
	#for i in range(0, label_len):
	#	dp['label-prefix'].append([ 0 ] * dp_token_len)
	#	dp['label-postfix'].append([ 0 ] * dp_token_len)
	#dp['case'] = 0



	label_begin = 0
	while less_than(token_list[label_begin]['position'], query['begin']):
		label_begin = label_begin + 1

	label_end = label_begin 
	while less_than(token_list[label_end]['position'], query['end']):
		label_end = label_end + 1

	if label_end - label_begin + 1 > label_len:
		return None


	prefix_point = -1
	prefix_len = min(label_begin, dp_token_len) 
	begin_idx = label_begin - prefix_len
	if 0 < label_begin :
		p = dp_token_len - prefix_len
		for i in range(begin_idx, label_begin):
			dp['prefix'][p] = convert_token(token_list[i])
			#if check_identity(token_list[i], token_list[label_begin]) == 1:
			#	prefix_point = p 
			##dp['label-prefix'][0][p] = check_identity(token_list[i], token_list[label_idx])
			p = p + 1

	p = 0 
	for i in range(label_begin, min(label_end, len(token_list))):
		dp['label-type'][p] = convert_token(token_list[i])		
		#for j in range(begin_idx, label_idx):
		#	dp['label-prefix'][i][p] = check_identity(token_list[j], token_list[label_idx + p])
		p = p + 1
	
	postfix_point = -1
	if label_end < len(token_list):
		postfix_len = min(len(token_list) - label_end - 1, dp_token_len)
		p = 0
		for i in range(label_end, label_end + postfix_len):
			dp['postfix'][p] = convert_token(token_list[i])
			#if check_identity(token_list[i], token_list[label_idx]) == 1 and postfix_point == -1:
			#	postfix_point = p
			##dp['label-postfix'][0][p] = check_identity(token_list[i], token_list[label_idx])
			p = p + 1
	dp['postfix'].reverse()

	
	#if postfix_point != -1:
	#	postfix_point = dp_token_len - 1 - postfix_point 

	#if prefix_point != -1 or postfix_point != -1:
	#	if prefix_point < postfix_point:
	#		dp['label-postfix'][0][postfix_point] = 1 
	#	else:
	#		dp['label-prefix'][0][prefix_point] = 1

	#if dp['label-type'][0] == token_dict['@unk']:
	#	dp['case'] = 1
	#	for i in range(0, dp_token_len):
	#		if dp['label-prefix'][0][i] == 1:
	#			dp['case'] = 2
	#			break 
	#	for i in range(0, dp_token_len):
	#		if dp['label-postfix'][0][i] == 1:
	#			dp['case'] = 2
	#			break 
	return dp


def print_datapoint (dp):
	global token_list

	print("{", end='')
	print("\"prefix\":" + str(dp['prefix']), end=', ')
	print("\"postfix\": " + str(dp['postfix']), end=', ') 
	print("\"label-type\": " + str(dp['label-type']), end='')
	print("}", end='\n') 
	#print("\"label-prefix\": " + str(dp['label-prefix']), end=', ')
	#print("\"label-postfix\": " + str(dp['label-postfix']), end=', ')
	#print("\"case\": " + str(dp['case']) + "}", end='\n') 


def main (target):
	global token_list, query_list, token_dict, dp_token_len, label_len




	argv = ['(',  'clang', '-fsyntax-only', '-Xclang', '-dump-tokens' ] 
	argv = argv + [target, '2>', '_tokens', ')']
	cmd = '' 
	for a in argv:
		cmd = cmd + ' ' + a
	os.system(cmd)
	os.system("grep \"" + target + "\" _tokens > " + "tokens")


	token_file = 'tokens'

	load_dictionary("token_dict") 
	read_tokens(token_file)
	read_quries()

	for q in query_list:
		dp = generate_datapoint(q)
		print_datapoint(dp)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Invalid input.")
		sys.exit(1)

	main(sys.argv[1])

