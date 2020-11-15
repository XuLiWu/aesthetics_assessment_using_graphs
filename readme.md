### Create and activate a virtual environment
`virtualenv -p /usr/bin/python3.6 venv_graph_aesthetics
source venv_graph_aesthetics/bin/activate
`
### Set up the virtual environment
`pip install -r requirements.txt`

### Extract Features
`sh extract_graph.sh`

### Train GNN on the features
`sh train.sh`

**Check opts_extractor.py and opts_train.py for all the hyperparameters**

### To view the results
`tensorboard --logdir dump/visuals/`