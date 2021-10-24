import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)#.to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=0.5, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape)#.to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y



class BaseGraphLearner(nn.Module):
    """  
    static_graph: idx as input + sampling
    dynamic_graph: raw time series input + topK
    """
    def __init__(self, num_nodes, node_emb_dim, graph_hidden):
        super(BaseGraphLearner, self).__init__()

        self.num_nodes = num_nodes
        self._graph_hidden = graph_hidden


        self.emb1 = nn.Embedding(self.num_nodes, node_emb_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_emb_dim)
        self.lin1 = nn.Linear(node_emb_dim, node_emb_dim)
        self.lin2 = nn.Linear(node_emb_dim, node_emb_dim)
            
        self.mlp = nn.ModuleList()
        graph_hidden = [1] + graph_hidden
        for i in range(len(graph_hidden)-1):
            self.mlp.append(nn.Linear(graph_hidden[i], graph_hidden[i+1]))


    def forward(self, idx):
        """  
        input: idx
        output: adj_mats
        """

       
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        nodevec1 = self.lin1(nodevec1)
        nodevec2 = self.lin2(nodevec2)

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adjs = torch.unsqueeze(a, 2)
        for i in range(len(self._graph_hidden)):
            adjs = self.mlp[i](adjs)

        return adjs
        
