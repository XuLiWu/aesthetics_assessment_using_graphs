import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SplineConv, GATConv, Set2Set, GlobalAttention, radius_graph, NNConv,  GraphSizeNorm
import pdb

class Net_GAT_Conv_SoA_1(torch.nn.Module):
    # Network performs better than SOA. Experiment 83.08, 86.02
    def __init__(self, args, dset_classes, in_features = 2048, hidden = 4096, heads = 8, out_features = 3072, edge_dim = 2):
        super(Net_GAT_Conv_SoA_1, self).__init__()
        # pdb.set_trace()
        self.lin01 = nn.Sequential(nn.Linear(in_features, 2048, bias = False), nn.ReLU(), \
                                   nn.BatchNorm1d(2048, eps=0.1), nn.Dropout(0.5))
        self.conv1 = GATConv(2048, 64, heads=8, dropout=0.0)
        self.conv2 = GATConv(2048, 64, heads=8, dropout=0.0)

        self.set2set = Set2Set(1024, processing_steps = 2, num_layers = 2)
        self.lin2 = nn.Sequential(nn.Linear(2048, 1024, bias = False),nn.ReLU(),\
                                  nn.BatchNorm1d(1024, eps=0.1), nn.Dropout(0.5))

        self.lin3 = nn.Linear(1024, 10)

    def forward(self, data, phase):
        # pdb.set_trace()
        data = data[1]
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.pos, data.batch
        edge_index = radius_graph(pseudo.float(), r=4, batch=batch)
        x = self.lin01(x)

        x_g1 = F.relu(self.conv1(x, edge_index))
        x_g2 = F.relu(self.conv2(x, edge_index))
        x = torch.cat([x_g1, x_g2], dim = 1)

        x = self.set2set(x, batch)

        x_last = self.lin2(x)
        x = self.lin3(x_last)
        return {'A2': x}

    def set_training_mode(self, phase, epoch):
        # if phase == 'train':
        if phase == 'train':
            # self.freeze_layers(False)
            self.train(True)
        else:
            # self.freeze_layers(True)
            self.eval()
