import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_op(op_name, layer_index, in_channels, out_channels, setting):
    name2op = {
        'Zero': lambda: Zero(),
        'Identity': lambda: Identity(),
        'Conv': lambda: Conv(in_channels, out_channels, **setting),
        'TemporalConv': lambda: TemporalConv(in_channels, out_channels, **setting),
        'SpatialGraphConv': lambda: SpatialGraphConv(in_channels, out_channels, **setting),
        'TemporalAttention': lambda: TemporalAttention(layer_index, in_channels, out_channels, **setting),
        'SpatialAttention': lambda: SpatialAttention(layer_index, in_channels, out_channels, **setting),
    }
    op = name2op[op_name]()
    return op


class MLP(nn.Module):
    def __init__(self, in_hidden, hiddens):
        super(MLP, self).__init__()
        self._in_hidden = in_hidden
        self._hiddens = hiddens
        self._layers = nn.ModuleList()
        for i, h in enumerate(hiddens):
            self._layers += [nn.Linear(in_hidden if i == 0 else hiddens[i - 1], h)]
            if i != len(hiddens) - 1:
                self._layers += [nn.ReLU()]

    def forward(self, x):
        for i, l in enumerate(self._layers):
            x = l(x)
        return x

class BasicOp(nn.Module):
    def __init__(self, **kwargs):
        super(BasicOp, self).__init__()

    def forward(self, inputs, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        cfg = []
        for (key, value) in self.setting:
            cfg += [str(key) + ': ' + str(value)]
        return str(self.type) + '(' + ', '.join(cfg) + ')'

    @property
    def type(self):
        raise NotImplementedError

    @property
    def setting(self):
        raise NotImplementedError


class Identity(BasicOp):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs, **kwargs):

        return inputs

    @property
    def type(self):
        return 'identity'

    @property
    def setting(self):
        return []


class Zero(BasicOp):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, inputs, **kwargs):

        return torch.zeros_like(inputs)

    @property
    def type(self):
        return 'zero'

    @property
    def setting(self):
        return []


class Conv(BasicOp):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=1, use_bn=True, dropout=0.3,
                 type_name='conv'):
        super(Conv, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._type_name = type_name
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._use_bn = use_bn
        self._dropout = dropout
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation)
        if use_bn: self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, **kwargs):

        x = torch.relu(inputs)

        x = self._conv(x)
        if self._use_bn: x = self._bn(x)
        if self._dropout > 0: x = F.dropout(x, self._dropout, training=self.training)
        return x

    @property
    def type(self):
        return self._type_name

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('out_channels', self._out_channels),
            ('kernel_size', self._kernel_size),
            ('stride', self._stride),
            ('padding', self._padding),
            ('dilation', self._dilation),
            ('use_bn', self._use_bn),
            ('dropout', self._dropout)
        ]


class TemporalConv(BasicOp):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=1, use_bn=True, dropout=0.3,
                 type_name='tc'):
        super(TemporalConv, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels


        self._type_name = type_name
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._use_bn = use_bn
        self._dropout = dropout

        self._conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self._conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        
        if use_bn: self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, **kwargs):

        x = inputs
        # x = torch.relu(x)
        
        filter = self._conv1(x)
        filter = torch.tanh(filter)
        gate = self._conv2(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        if self._dropout > 0: x = F.dropout(x, self._dropout, training=self.training)
        return x

    @property
    def type(self):
        return self._type_name

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('out_channels', self._out_channels),
            ('kernel_size', self._kernel_size),
            ('stride', self._stride),
            ('padding', self._padding),
            ('dilation', self._dilation),
            ('use_bn', self._use_bn),
            ('dropout', self._dropout)
        ]


def gconv(x, A):
    x = torch.einsum('ncvl,vw->ncwl', (x, A))
    return x.contiguous()

def dgconv(x, A):
    x = torch.einsum('ncvl,nvw->ncwl', (x, A))
    return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class SpatialGraphConv(BasicOp):
    def __init__(self, in_channels, out_channels,
                 input_graph_hidden, graph_hiddens, num_graphs, order, k=20, alpha=0.05, use_bn=True, dropout=0.3):
        super(SpatialGraphConv, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._input_graph_hidden = input_graph_hidden
        self._graph_hiddens = graph_hiddens

        self._num_graphs = num_graphs
        self._order = order
        self._use_bn = use_bn
        self._dropout = dropout
        self._k = k
        self._alpha = alpha

        self._edge_mlp = MLP(input_graph_hidden, graph_hiddens + [num_graphs])
        self._linear1 = nn.Conv2d(in_channels * (num_graphs * order + 1), out_channels, kernel_size=(1, 1),
                                 stride=(1, 1))
        self._linear2 = nn.Conv2d(in_channels * (num_graphs * order + 1), out_channels, kernel_size=(1, 1),
                                 stride=(1, 1))
        if use_bn: self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, adj_mats, **kwargs):

        adj_mats = self._edge_mlp(adj_mats)
        
        adjs = []

        
        for num in range(self._num_graphs):
            mask = torch.zeros(adj_mats.shape[0], adj_mats.shape[1]).to(device)
            mask.fill_(float('0'))
            adj = adj_mats[:,:,num].squeeze()
            adj = torch.softmax(adj, dim=1)
            s1,t1 = adj.topk(int(self._k),1)
            mask.scatter_(1,t1,s1.fill_(1))
            adj = adj*mask
            
            adj = (1.0 / (adj.sum(dim=1, keepdim=True) + 1e-8)) * adj
            adjs.append(adj)
        adj_mats = adjs

        x = torch.relu(inputs)

        outputs1 = [x]
        for i in range(self._num_graphs):
            y = x
            for j in range(self._order):
                y = self._alpha * x + (1 - self._alpha) * gconv(y, adj_mats[i])
                outputs1 += [y]

        x1 = torch.cat(outputs1, dim=1)
        x1 = self._linear1(x1)

        # outputs2 = [x]
        # for i in range(self._num_graphs):
        #     adj1 = torch.softmax(adj_mats[i].transpose(1,0), dim=1)
        #     y = x
        
        #     for j in range(self._order):
        #         y = self._alpha * x + (1 - self._alpha) * gconv(y, adj1)
        #         outputs2 += [y]
        # x2 = torch.cat(outputs2, dim=1)
        # x2 = self._linear1(x2)

        # x = x1 + x2
        x = x1

        if self._use_bn: x = self._bn(x)
        if self._dropout > 0: x = F.dropout(x, self._dropout, training=self.training)
        return x

    @property
    def type(self):
        return 'sc'

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('out_channels', self._out_channels),
            ('graph_hiddens', self._graph_hiddens),
            ('num_graphs', self._num_graphs),
            ('order', self._order),
            ('use_bn', self._use_bn),
            ('dropout', self._dropout)
        ]


class TemporalAttention(BasicOp):
    def __init__(self, layer_index, in_channels, out_channels, num_nodes, in_length, use_bn=True, dropout=0.3, type_name='tatt'):
        super(TemporalAttention, self).__init__()
        self._in_channels = in_channels
        self._num_nodes = num_nodes
        self._type_name = type_name
        self._in_length = in_length[layer_index-1]
        self._dropout = dropout
        self._use_bn = use_bn

        self.U1 = nn.Parameter(torch.cuda.FloatTensor(num_nodes))
        self.U2 = nn.Parameter(torch.cuda.FloatTensor(in_channels, num_nodes))
        self.U3 = nn.Parameter(torch.cuda.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.cuda.FloatTensor(1, self._in_length, self._in_length))
        self.Ve = nn.Parameter(torch.cuda.FloatTensor(self._in_length, self._in_length))

        if use_bn: self._bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs, **kwargs):
        '''
        :param inputs: (batch_size, N, F_in, T)
        :return E_normalized: (B, T, T) 
                TAtt_output: (batch_size, N, F_in, T)
        '''

        x = torch.relu(inputs)

        inputs = torch.transpose(x, 1, 2)

        batch_size, num_nodes, num_of_features, in_length = inputs.shape

        lhs = torch.matmul(torch.matmul(inputs.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, inputs)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        TAtt_output = torch.matmul(inputs.reshape(batch_size, -1, in_length), E_normalized).reshape(batch_size, num_nodes, num_of_features, in_length)

        TAtt_output = torch.transpose(TAtt_output, 1, 2)
        # if self._use_bn: TAtt_output = self._bn(TAtt_output)
        if self._dropout > 0: TAtt_output = F.dropout(TAtt_output, self._dropout, training=self.training)

        return TAtt_output
               

    @property
    def type(self):
        return self._type_name

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('num_nodes', self._num_nodes),
            ('in_length', self._in_length),
            ('dropout', self._dropout)
        ]


class SpatialAttention(BasicOp):
    '''
    compute spatial attention scores
    '''
    def __init__(self, layer_index, in_channels, out_channels, num_nodes, in_length, gdep=3, alpha=0.05, use_bn=True, dropout=0.3, type_name='satt'):
        super(SpatialAttention, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_nodes = num_nodes
        self._type_name = type_name
        self._in_length = in_length[layer_index-1]
        self._gdep = gdep
        self._alpha = alpha
        self._dropout = dropout
        self._use_bn = use_bn

        self.W1 = nn.Parameter(torch.cuda.FloatTensor(self._in_length))
        self.W2 = nn.Parameter(torch.cuda.FloatTensor(in_channels, self._in_length))
        self.W3 = nn.Parameter(torch.cuda.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.cuda.FloatTensor(1, num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.cuda.FloatTensor(num_nodes, num_nodes))

        self.gconv = dy_mixprop(in_channels, out_channels, gdep, dropout, alpha)
        
        if use_bn: self._bn = nn.BatchNorm2d(out_channels)


    def forward(self, inputs, **kwargs):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: S_normalized: (B,N,N)
        '''

        x = torch.relu(inputs)

        inputs = torch.transpose(x, 1, 2)
        
        
        lhs = torch.matmul(torch.matmul(inputs, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, inputs).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)


        SAtt_output = self.gconv(x, S_normalized)

        # if self._use_bn: SAtt_output = self._bn(SAtt_output)
        if self._dropout > 0: SAtt_output = F.dropout(SAtt_output, self._dropout, training=self.training)

        return SAtt_output
    
    @property
    def type(self):
        return self._type_name

    @property
    def setting(self):
        return [
            ('in_channels', self._in_channels),
            ('out_channels', self._out_channels),
            ('num_nodes', self._num_nodes),
            ('in_length', self._in_length),
            ('gdep', self._gdep),
            ('alpha', self._alpha),
            ('dropout', self._dropout)
        ]


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_nodes, in_channels, in_length = x.shape

        outputs = []

        for time_step in range(in_length):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_nodes, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class dgconv_mixhop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dgconv_mixhop, self).__init__()

        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):

        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*gconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()

        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x, adj):
        
        h = x

        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*dgconv(h,adj)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)

        adj1 = torch.softmax(adj.transpose(2,1), dim=2)
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * dgconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2
