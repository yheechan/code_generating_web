import torch

def predict(
    prefix,
    postfix,
    label=[2],
    model=None,
    device=None
):

	# set model to predict mode
	model.to(device)
	model.eval()

	# ********************* INSTANTIATE MODEL INPUT DATA *********************

	# [batch_size (1), token_length (64)]
	prefix = torch.tensor(prefix).unsqueeze(dim=0)
	postfix = torch.tensor(postfix).unsqueeze(dim=0)
	label = torch.tensor(label).unsqueeze(dim=0)

	prefix = prefix.to(device)
	postfix = postfix.to(device)
	label = label.to(device)


	# usage to save likelihoods of prefix and postfix as a whole
	# [batch_size, 64] --> [token len (128), batch_size, 1 (1 or 0)]
	total_labels = torch.zeros((prefix.shape[1]*2, prefix.shape[0], 1)).to(device)




	# ********************* PREDICT TOKEN SEQUENCE *********************

	# [label_len (128 labels), batch_size (1000), single likelihood (BCE -1)]
	results = model(prefix, postfix, label)



	# ********** SPLIT PREDICTED LIKELIHOOD **********

	# torch.set_printoptions(sci_mode=False, precision=20)
	# [prefix prediction, postfix prediction] in likelihood
	prefix_likelihood, postfix_likelihood = torch.split(results, 64)

	# [batch_size (1000), token_len (64)]
	prefix_likelihood = prefix_likelihood.permute(1, 0, 2).squeeze()
	postfix_likelihood = postfix_likelihood.permute(1, 0, 2).squeeze()



	# change torch tensor to like
	prefix_likelihood_list = prefix_likelihood.tolist()
	postfix_likelihood_list = postfix_likelihood.tolist()



	return prefix_likelihood_list, postfix_likelihood_list
