import os
from nc_dataset import NodeClassificationDataset, NodeClassificationEvaluator

data_path = '../data' # replace this with the path where you save the datasets
dataset_name = 'ele-fashion'
feat_name = 'imagebind'
verbose = True
device = 'cpu' # use 'cuda' if GPU is available

dataset = NodeClassificationDataset(
	root=os.path.join(data_path, dataset_name),
	feat_name=feat_name,
	verbose=verbose,
	device=device
)

graph = dataset.graph
# type(graph) would be dgl.DGLGraph
# use graph.ndata['feat'] to get the features
# use graph.ndata['label'] to get the labels (i.e., classes)
# use graph.ndata['train_mask'], graph.ndata['val_mask'], and graph.ndata['test_mask'] to get the corresponding masks

graph

#########################

eval_metric = 'rocauc' # 'acc' is also supported
evaluator = NodeClassificationEvaluator(eval_metric=eval_metric)
# use evaluator.expected_input_format and evaluator.expected_output_format to see the details about the format

input_dict = {'y_true': ..., 'y_pred': ...} # get input_dict using the model you trained
result = evaluator.eval(input_dict=input_dict)