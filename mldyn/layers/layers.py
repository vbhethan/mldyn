import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# See https://github.com/ethanfetaya/NRI/ and https://github.com/loeweX/AmortizedCausalDiscovery/


class MLP(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out) 
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.1)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def batch_norm(self, inputs):
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.bn(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        
        def forward(self, x):
            x = F.elu(self.fc1(x))
            x = F.droupout(x, self.dropout_prob, training=self.training)
            x = F.elu(self.fc2(x)) 
            return self.batch_norm(x)
        
class MLPEncoder(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLPEncoder, self).__init__()

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)

        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)
    
    def node2edge(self, x, rel_rec, rel_send):
        send = torch.matmul(rel_send, x)
        rec = torch.matmul(rel_rec, x)
        edges = torch.cat([send, rec], dim=2)
        return edges
    
    def forward(self, inputs, rel_rec, rel_send):
        # Inputs will have shape [num_sims, num_particles, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape [num_sims, num_particles, num_timesteps * num_dims]
        x = self.mlp1(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        # Implement the factor graph to start 
        x = self.edge2node(x, rel_rec, rel_send)
        x = self.mlp3(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = torch.cat([x, x_skip], dim=2)
        x = self.mlp4(x)

        return self.fc_out(x)


class MLPDecoder(nn.Module):
    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid, do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)]
        )
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send, single_timestep_rel_type):
        # single_timesteps_inputs has shape [batch_size, num_timesteps, num_particles, num_dims]
        # single_timestep_rel_type has shape [batch_size, num_timesteps, num_particles*(num_particles-1), num_edge_types]

        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)
        