import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nas.candidate_op import BasicOp, create_op
from utils.helper import Mode


class MixedOp_1(BasicOp):
    def __init__(self, layer_index, in_channels, out_channels, candidate_op_profiles_1):
        super(MixedOp_1, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        
        self.layer_index = layer_index
        self._num_ops = len(candidate_op_profiles_1)
        self._candidate_op_profiles = candidate_op_profiles_1
        self._candidate_ops = nn.ModuleList()
        for (op_name, profile) in self._candidate_op_profiles:
            self._candidate_ops += [create_op(op_name, self.layer_index, self._in_channels, self._out_channels, profile)]
        
        self._candidate_alphas = nn.Parameter(torch.zeros(self._num_ops), requires_grad=True)
        self.set_mode(Mode.NONE)

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.NONE:
            self._sample_idx = None
        elif mode == Mode.ONE_PATH_FIXED:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            op = torch.argmax(probs).item()
            self._sample_idx = np.array([op], dtype=np.int32)
        elif mode == Mode.ONE_PATH_RANDOM:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(probs, 1, replacement=True).cpu().numpy()
        elif mode == Mode.TWO_PATHS:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(probs, 2, replacement=True).cpu().numpy()
        elif mode == Mode.ALL_PATHS:
            self._sample_idx = np.arange(self._num_ops)

    def forward(self, inputs, adj_mats):
        
        probs = F.softmax(self._candidate_alphas[self._sample_idx], dim=0)
        output = 0
        for i, idx in enumerate(self._sample_idx):
            output += probs[i] * self._candidate_ops[idx](inputs, adj_mats=adj_mats)
        return output

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p

    def num_weight_parameters(self):
        from utils.helper import num_parameters
        counter = 0
        for idx in self._sample_idx:
            counter += num_parameters(self._candidate_ops[idx])
        return counter

    def __repr__(self):
        # mode info
        out_str = ''
        out_str += 'mode: ' + str(self._mode) + str(self._sample_idx) + ',\n'
        # probability of each op & its info
        probs = F.softmax(self._candidate_alphas.data, dim=0)
        for i in range(self._num_ops):
            out_str += 'op:%d, prob: %.3f, info: %s,' % (i, probs[i].item(), self._candidate_ops[i])
            if i + 1 < self._num_ops:
                out_str += '\n'

        from utils.helper import add_indent
        out_str = 'mixed_op {\n%s\n}' % add_indent(out_str, 4)
        return out_str

    def render_name(self):
        probs = F.softmax(self._candidate_alphas.data, dim=0)
        index = torch.argmax(probs).item()
        out_str = self._candidate_ops[index].type
        out_str = '%s(%.2f)' % (out_str, probs[index])
        return out_str




class MixedOp_2(BasicOp):
    def __init__(self, layer_index, in_channels, out_channels, candidate_op_profiles_2):
        super(MixedOp_2, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.layer_index = layer_index
        self._num_ops = len(candidate_op_profiles_2)
        self._candidate_op_profiles = candidate_op_profiles_2
        self._candidate_ops = nn.ModuleList()
        for (op_name, profile) in self._candidate_op_profiles:
            self._candidate_ops += [create_op(op_name, self.layer_index, self._in_channels, self._out_channels, profile)]
        
        self._candidate_alphas = nn.Parameter(torch.zeros(self._num_ops), requires_grad=True)
        self.set_mode(Mode.NONE)

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.NONE:
            self._sample_idx = None
        elif mode == Mode.ONE_PATH_FIXED:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            op = torch.argmax(probs).item()
            self._sample_idx = np.array([op], dtype=np.int32)
        elif mode == Mode.ONE_PATH_RANDOM:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(probs, 1, replacement=True).cpu().numpy()
        elif mode == Mode.TWO_PATHS:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(probs, 2, replacement=True).cpu().numpy()
        elif mode == Mode.ALL_PATHS:
            self._sample_idx = np.arange(self._num_ops)

    def forward(self, inputs, adj_mats):
        probs = F.softmax(self._candidate_alphas[self._sample_idx], dim=0)
        output = 0
        for i, idx in enumerate(self._sample_idx):
            output += probs[i] * self._candidate_ops[idx](inputs, adj_mats=adj_mats)
        return output

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p
                
    def num_weight_parameters(self):
        from utils.helper import num_parameters
        counter = 0
        for idx in self._sample_idx:
            counter += num_parameters(self._candidate_ops[idx])
        return counter

    def __repr__(self):
        # mode info
        out_str = ''
        out_str += 'mode: ' + str(self._mode) + str(self._sample_idx) + ',\n'
        # probability of each op & its info
        probs = F.softmax(self._candidate_alphas.data, dim=0)
        for i in range(self._num_ops):
            out_str += 'op:%d, prob: %.3f, info: %s,' % (i, probs[i].item(), self._candidate_ops[i])
            if i + 1 < self._num_ops:
                out_str += '\n'

        from utils.helper import add_indent
        out_str = 'mixed_op {\n%s\n}' % add_indent(out_str, 4)
        return out_str

    def render_name(self):
        probs = F.softmax(self._candidate_alphas.data, dim=0)
        index = torch.argmax(probs).item()
        out_str = self._candidate_ops[index].type
        out_str = '%s(%.2f)' % (out_str, probs[index])
        return out_str
