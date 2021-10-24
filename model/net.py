import torch
import torch.nn as nn
import math

from model.nas.cell import STCell
from model.base_graph_learner import BaseGraphLearner
from model.multi_scale_block import MultiScale
from utils.helper import Mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_layer(name, layer_index, hidden_channels, num_layer_per_cell, candidate_op_profiles_1, candidate_op_profiles_2):
    if name == 'STCell':
        return STCell(layer_index, hidden_channels, num_layer_per_cell, candidate_op_profiles_1, candidate_op_profiles_2)
    if name == 'ConvPooling':
        return nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(1, 3), padding=(0, 1),
                         stride=(1, 2))
    if name == 'AvgPooling':
        return nn.AvgPool2d(kernel_size=(1,3), padding=(0,1), stride=(1,2))
    raise Exception('unknown layer name!')


class Net(nn.Module):

    def __init__(self,
                 in_length, out_length, num_nodes, node_emb_dim, graph_hidden, 
                 in_channels, out_channels, hidden_channels, scale_channels, end_channels,
                 layer_structure,
                 num_layer_per_cell, candidate_op_profiles_1, candidate_op_profiles_2):
        super(Net, self).__init__()

        self._in_length = in_length
        self._out_length = out_length
        self._num_nodes = num_nodes
        self._node_emb_dim = node_emb_dim
        self._graph_hidden = graph_hidden

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._hidden_channels = hidden_channels
        self._scale_channels = scale_channels
        self._end_channels = end_channels

        self._layer_structure = layer_structure
        self._scales = len(layer_structure)

        self._num_layer_per_cell = num_layer_per_cell
        self._candidate_op_profiles_1 = candidate_op_profiles_1
        self._candidate_op_profiles_2 = candidate_op_profiles_2

        
        self.idx = torch.arange(self._num_nodes).to(device=device)
        self._base_graph_learner = BaseGraphLearner(self._num_nodes, self._node_emb_dim, self._graph_hidden)

        self._STCells = nn.ModuleList()
        self._scale_convs = nn.ModuleList()
        # self._pooling = nn.ModuleList()
        # for i in range(2):
        #     self._pooling.append(nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(1, 3), padding=(0, 1),
        #                  stride=(1, 2)))

        self._multi_scale = MultiScale(self._scales, hidden_channels, hidden_channels)
        
        layer_index = 0
        scale_index = 0
        for scale in self._layer_structure:
            for Cell in scale:
                layer_index += 1
                self._STCells += [STCell(layer_index, hidden_channels, num_layer_per_cell, candidate_op_profiles_1, candidate_op_profiles_2)]
                self._scale_convs += [nn.Conv2d(self._hidden_channels, self._scale_channels, kernel_size=(1, int(self._in_length/math.pow(2, scale_index))))]
            scale_index += 1

        self._start_conv = nn.Conv2d(self._in_channels, self._hidden_channels, kernel_size=(1, 1))
        self._scale0 = nn.Conv2d(self._hidden_channels, self._scale_channels, kernel_size=(1, self._in_length), bias=True)
        self._end_conv1 = nn.Conv2d(self._scale_channels, self._end_channels, kernel_size=(1, 1))
        self._end_conv2 = nn.Conv2d(self._end_channels, self._out_channels * self._out_length, kernel_size=(1, 1))

        
        self.set_mode(Mode.NONE)
        

    def forward(self, x, mode):
        
        self.set_mode(mode)


        adj_mats = self._base_graph_learner(self.idx)

        x = self._start_conv(x)
        scale_output = self._scale0(x)
        
        scale = self._multi_scale(x)
        
        layer_index = 0

        for i in range(self._scales):
            x1 = scale[i]
            for cell_index in range(len(self._layer_structure[i])):
                x1 = self._STCells[layer_index](x1, adj_mats) 
                scale_output = scale_output + self._scale_convs[layer_index](x1)
                layer_index += 1
        
        # layer_index = 0
        # x1 = x
        # for i in range(self._scales):
            
        #     for cell_index in range(len(self._layer_structure[i])):
        #         x1 = self._STCells[layer_index](x1, adj_mats) 
        #         scale_output = scale_output + self._scale_convs[layer_index](x1)
        #         layer_index += 1
        #     if i < self._scales-1:
        #         x1 = self._pooling[i](x1)
            

        x = torch.relu(scale_output)
        x = self._end_conv1(x)
        x = torch.relu(x)
        x = self._end_conv2(x)
        x = x.view(x.size(0), self._out_channels, self._out_length, x.size(2))
        x = x.transpose(2, 3).contiguous()  # [b, c, n, t]

        self.set_mode(Mode.NONE)
        return x

    def set_mode(self, mode):
        self._mode = mode
        for l in self._STCells:
            if isinstance(l, STCell):
                l.set_mode(mode)

    def weight_parameters(self):
        for m in [self._base_graph_learner, self._multi_scale, self._start_conv, self._end_conv1, self._end_conv2, self._scale0]:
            for p in m.parameters():
                yield p
        for m in self._scale_convs:
            for p in m.parameters():
                yield p
        # for m in self._pooling:
        #     for p in m.parameters():
        #         yield p
        for m in self._STCells:
            if isinstance(m, STCell):
                for p in m.weight_parameters():
                    yield p

    def arch_parameters(self):
        for m in self._STCells:
            if isinstance(m, STCell):
                for p in m.arch_parameters():
                    yield p

    def num_weight_parameters(self):
        from utils.helper import num_parameters
        current_mode = self._mode
        self.set_mode(Mode.ONE_PATH_FIXED)
        count = 0
        for m in [self._base_graph_learner, self._multi_scale, self._start_conv, self._end_conv1, self._end_conv2, self._scale0]:
            count += num_parameters(m)
        for m in self._scale_convs:
            count += num_parameters(m)
        # for m in self._pooling:
        #     count += num_parameters(m)
        for m in self._STCells:
            if isinstance(m, STCell):
                count += m.num_weight_parameters()

        self.set_mode(current_mode)
        return count

    def __repr__(self):
        out_str = []
        for l in self._STCells:
            out_str += [str(l)]

        from utils.helper import add_indent
        out_str = 'NAS {\n%s\n}\n' % add_indent('\n'.join(out_str), 4)
        return out_str
