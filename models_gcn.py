import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pdb
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, act=nn.ReLU(), normalize_input=True
    ):
        super(MLP, self).__init__()

        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.act = act
        self.normalize_input = normalize_input

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # x = F.normalize(x, p=2, dim=-1)
        if self.normalize_input:
            x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
        x = self.act(self.linear_1(x))
        return self.linear_2(x)


# # GCN basic operation
class GraphConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        normalize_embedding=False,
        normalize_embedding_l2=False,
        att=False,
        mpnn=False,
        graphsage=False,
    ):
        super(GraphConv, self).__init__()
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_l2 = normalize_embedding_l2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.att = att
        self.mpnn = mpnn
        self.graphsage = graphsage

        if self.graphsage:
            self.out_compute = MLP(
                input_dim=input_dim * 2,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                act=nn.ReLU(),
                normalize_input=False,
            )
        elif self.mpnn:
            self.out_compute = MLP(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                act=nn.ReLU(),
                normalize_input=False,
            )
        else:
            self.out_compute = MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                act=nn.ReLU(),
                normalize_input=False,
            )
        if self.att:
            self.att_compute = MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                act=nn.LeakyReLU(0.2),
                normalize_input=False,
            )
        if self.mpnn:
            self.mpnn_compute = MLP(
                input_dim=input_dim * 2,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                act=nn.ReLU(),
                normalize_input=False,
            )

        # self.W = nn.Parameter(torch.zeros(size=(input_dim, input_dim)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        if self.att:
            x_att = self.att_compute(x)
            # pdb.set_trace()
            att = x_att @ x_att.permute(1, 0)
            # pdb.set_trace()
            att = self.softmax(att)
            # pdb.set_trace()
            pred = torch.matmul(adj * att, x)
            # pdb.set_trace()
        elif self.mpnn:
            x1 = x.unsqueeze(0).repeat(x.shape[0], 1, 1)
            # x2 = x1.permute(1,0,2)
            x2 = x.unsqueeze(1).repeat(1, x.shape[0], 1)
            e = torch.cat((x1, x2), dim=-1)
            e = self.mpnn_compute(e)
            pred = torch.mean(adj.unsqueeze(-1) * e, dim=1)
            # return pred
        else:
            pred = torch.matmul(adj, x)
        # pdb.set_trace()
        if self.graphsage:
            pred = torch.cat((pred, x), dim=-1)

        pred = self.out_compute(pred)
        # pdb.set_trace()
        if self.normalize_embedding:
            pred = (pred - torch.mean(pred, dim=0)) / torch.std(pred, dim=0)
        if self.normalize_embedding_l2:
            pred = F.normalize(pred, p=2, dim=-1)
        # pdb.set_trace()
        return pred


class GCN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers=2,
        concat=False,
        normalize_embedding=True,
        normalize_embedding_l2=False,
        att=False,
        mpnn=False,
        graphsage=False,
    ):
        super(GCN, self).__init__()
        self.concat = concat
        self.att = att
        self.num_layers = num_layers
        self.conv_first = GraphConv(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            normalize_embedding=normalize_embedding,
            normalize_embedding_l2=normalize_embedding_l2,
            att=att,
            mpnn=mpnn,
            graphsage=graphsage,
        )

        if self.num_layers > 1:
            self.conv_block = nn.ModuleList(
                [
                    GraphConv(
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        normalize_embedding=normalize_embedding,
                        normalize_embedding_l2=normalize_embedding_l2,
                        att=att,
                        mpnn=mpnn,
                        graphsage=graphsage,
                    )
                    for i in range(num_layers - 2)
                ]
            )

            self.conv_last = GraphConv(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                normalize_embedding=normalize_embedding,
                normalize_embedding_l2=normalize_embedding_l2,
                att=att,
                mpnn=mpnn,
                graphsage=graphsage,
            )
        if self.concat:
            self.MLP = MLP(
                input_dim=hidden_dim * num_layers,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                act=nn.ReLU(),
                normalize_input=True,
            )
        else:
            self.MLP = MLP(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                act=nn.ReLU(),
                normalize_input=True,
            )
        self.act = nn.ReLU()
        self.w = nn.Parameter(torch.zeros([1]))
        self.w.data = nn.init.constant_(self.w, 1)
        self.b = nn.Parameter(torch.zeros([1]))
        self.b.data = nn.init.constant_(self.b, 0)

    def forward(self, x, adj):
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.num_layers > 1:
            for i in range(len(self.conv_block)):
                x = self.conv_block[i](x, adj)
                x = self.act(x)
            x = self.conv_last(x, adj)
        return x