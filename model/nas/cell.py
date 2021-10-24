import torch.nn as nn

from model.nas.mixed_op import MixedOp_1, MixedOp_2
from utils.helper import Mode


class Cell(nn.Module):
    def __init__(self, layer_index, channels, num_mixed_ops, candidate_op_profiles_1, candidate_op_profiles_2):
        super(Cell, self).__init__()
        # create mixed operations
        self._channels = channels
        self._num_mixed_ops = num_mixed_ops
        self.layer_index = layer_index
        self._mixed_ops_1 = nn.ModuleList()
        self._mixed_ops_2 = nn.ModuleList()
        for i in range(self._num_mixed_ops):
            self._mixed_ops_1 += [MixedOp_1(self.layer_index, self._channels, self._channels, candidate_op_profiles_1)]
            for ii in range(i):
                self._mixed_ops_2 += [MixedOp_2(self.layer_index, self._channels, self._channels, candidate_op_profiles_2)]

        self.set_mode(Mode.NONE)

    def set_mode(self, mode_1, mode_2=Mode.ONE_PATH_FIXED):
        # self._mode = mode_1
        mode_2 = mode_1
        for op in self._mixed_ops_1:
            op.set_mode(mode_1)
        for op in self._mixed_ops_2:
            op.set_mode(mode_2)

    def arch_parameters(self):
        index = 0
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops_1[i].arch_parameters():
                yield p
            
            for ii in range(i):
                for p in self._mixed_ops_2[index].arch_parameters():
                    yield p
                index += 1

    def weight_parameters(self):
        index = 0
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops_1[i].weight_parameters():
                yield p
            
            for ii in range(i):
                for p in self._mixed_ops_2[index].weight_parameters():
                    yield p
                index += 1

    def num_weight_parameters(self):
        count = 0
        index = 0
        for i in range(self._num_mixed_ops):
            count += self._mixed_ops_1[i].num_weight_parameters()
            for ii in range(i):
                count += self._mixed_ops_2[index].num_weight_parameters()
                index += 1
        return count

    def forward(self, x, adj_mats):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class STCell(Cell):
    def __init__(self, layer_index, channels, num_mixed_ops, candidate_op_profiles_1, candidate_op_profiles_2):
        super(STCell, self).__init__(layer_index, channels, num_mixed_ops, candidate_op_profiles_1, candidate_op_profiles_2)

    def forward(self, x, adj_mats):
        # calculate outputs
        node_idx = 0
        current_output = 0

        node_outputs = [x]
        for i in range(self._num_mixed_ops):
            for ii in range(i):
                current_output += self._mixed_ops_2[node_idx](node_outputs[ii], adj_mats)
                node_idx += 1

            current_output += self._mixed_ops_1[i](node_outputs[-1], adj_mats)
            node_outputs += [current_output]
            current_output = 0

        ret = 0
        for x in node_outputs[:]:
            ret = ret + x
        return ret

    def __repr__(self):
        edge_cnt = 0
        out_str = []
        index = 0
        for i in range(self._num_mixed_ops):
            out_str += ['mixed_op_1: %d\n%s' % (i, self._mixed_ops_1[i])]
            for ii in range(i):
                out_str += ['mixed_op_2: %d\n%s' % (ii, self._mixed_ops_2[index])]
                index += 1

        from utils.helper import add_indent
        out_str = 'STCell {\n%s\n}' % add_indent('\n'.join(out_str), 4)
        return out_str
