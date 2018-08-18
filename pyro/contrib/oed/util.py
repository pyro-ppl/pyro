import torch

def get_indices(labels, sizes=None, tensors=None):
	indices = []
	start = 0
	for label in labels:
		if sizes is not None:
			end = start+sizes[label][0]
		else:
			end = start+tensors[label].shape[0]
		indices.extend(range(start, end))
		start = end
	return torch.tensor(indices)
