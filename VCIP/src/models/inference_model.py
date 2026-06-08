import torch
import torch.nn as nn
from src.models.dynamic_model import DynamicParamNetwork

class InferenceModel(nn.Module):
    def __init__(self, config):
        super(InferenceModel, self).__init__()
        self.z_dim = config['model']['z_dim']
        self.hidden_dim = config['model']['inference']['hidden_dim']
        self.num_layers = config['model']['inference']['num_layers']
        self.do = config['model']['inference']['do']
        self.hiddens_F_mu = config['model']['inference']['hiddens_F_mu']
        self.hiddens_F_logvar = config['model']['inference']['hiddens_F_logvar']
        self.history_dim = config['model']['auxiliary']['hidden_dim']
        self.treatment_dim = config['dataset']['treatment_size']
        self.output_dim = config['dataset']['output_size']
        self.input_dim = self.history_dim + self.treatment_dim + self.output_dim + self.z_dim
        self.static_size = config['dataset']['static_size']
        self.autoregressive = config['dataset']['autoregressive']
        self.input_size = config['dataset']['input_size']
        self.treatment_size = config['dataset']['treatment_size']
        self.predict_X = config['dataset']['predict_X']
        self.treatment_hidden_dim = config['model']['generative']['treatment_hidden_dim']
        self.dropout = config['exp']['dropout']
        self.input_size = config['dataset']['input_size']
        

        if self.static_size > 0:
            input_size = self.input_size + self.static_size + self.treatment_size
        else:
            input_size = self.input_size + self.treatment_size
        if self.autoregressive:
            input_size += self.output_dim
        self.lstm_history = nn.LSTM(input_size, self.z_dim, self.num_layers, batch_first=True, dropout=self.dropout)
        
        # 存储隐状态
        self.hidden_state = None
        self.cell_state = None

        # input_size = self.z_dim + self.treatment_dim + self.output_dim
        input_size = self.z_dim * 2 + self.output_dim
        if self.do:
            input_size += self.treatment_hidden_dim
        print(f"input_size: {input_size}, self.do {self.do}")
        # input_size = self.z_dim + self.output_dim
        self.lstm = nn.LSTM(input_size, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout)
        
        # input_size = self.z_dim + self.treatment_size
        # input_size = self.z_dim + self.treatment_hidden_dim
        input_size = self.hidden_dim
        self.fc_mu = nn.Sequential()
        if -1 not in self.hiddens_F_mu:
            for i in range(len(self.hiddens_F_mu)):
                if i == 0:
                    self.fc_mu.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_F_mu[i]))
                else:
                    self.fc_mu.add_module('elu{}'.format(i), nn.ELU())
                    self.fc_mu.add_module('fc{}'.format(i), nn.Linear(self.hiddens_F_mu[i-1], self.hiddens_F_mu[i]))
            self.fc_mu.add_module('elu{}'.format(len(self.hiddens_F_mu)), nn.ELU())
            self.fc_mu.add_module('fc{}'.format(len(self.hiddens_F_mu)), nn.Linear(self.hiddens_F_mu[-1], self.z_dim))
        else:
            self.fc_mu.add_module('fc{}'.format(1), nn.Linear(input_size, self.z_dim))
        
        self.fc_logvar = nn.Sequential()
        if -1 not in self.hiddens_F_logvar:
            for i in range(len(self.hiddens_F_logvar)):
                if i == 0:
                    self.fc_logvar.add_module('fc{}'.format(i), nn.Linear(input_size, self.hiddens_F_logvar[i]))
                else:
                    self.fc_logvar.add_module('elu{}'.format(i), nn.ELU())
                    self.fc_logvar.add_module('fc{}'.format(i), nn.Linear(self.hiddens_F_logvar[i-1], self.hiddens_F_logvar[i]))
            self.fc_logvar.add_module('elu{}'.format(len(self.hiddens_F_logvar)), nn.ELU())
            self.fc_logvar.add_module('fc{}'.format(len(self.hiddens_F_logvar)), nn.Linear(self.hiddens_F_logvar[-1], self.z_dim))
        else:
            self.fc_logvar.add_module('fc{}'.format(1), nn.Linear(input_size, self.z_dim))

        input_dim = self.z_dim + self.output_dim
        hidden_dim = self.hidden_dim
        output_dim = self.hidden_dim
        self.transition_network = DynamicParamNetwork(input_dim, hidden_dim, output_dim, num_rbf_centers=5)

        self.predict_y_history = config['model']['inference']['predict_y_history']
        self.predict_y_history_net = nn.Sequential()
        input_size = self.z_dim + self.treatment_size
        if -1 not in self.predict_y_history:
            for i in range(len(self.predict_y_history)):
                if i == 0:
                    self.predict_y_history_net.add_module('fc{}'.format(i), nn.Linear(input_size, self.predict_y_history[i]))
                else:
                    self.predict_y_history_net.add_module('elu{}'.format(i), nn.ELU())
                    self.predict_y_history_net.add_module('fc{}'.format(i), nn.Linear(self.predict_y_history[i-1], self.predict_y_history[i]))
            self.predict_y_history_net.add_module('elu{}'.format(len(self.predict_y_history)), nn.ELU())
            self.predict_y_history_net.add_module('fc{}'.format(len(self.predict_y_history)), nn.Linear(self.predict_y_history[-1], self.output_dim))
        else:
            self.predict_y_history_net.add_module('fc{}'.format(1), nn.Linear(input_size, self.output_dim))
    
    def init_hidden(self, batch_size):
        """初始化LSTM隐状态"""
        device = next(self.parameters()).device
        self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        self.cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

    def build_H_t(self, H_t):
        if self.static_size > 0:
            if self.predict_X:
                x = H_t['vitals']
                x = torch.cat((x, H_t['static_features']), dim=-1)
            # when we don't predict x, we use static features as the current_covariates
            else:
                x = H_t['static_features']
        # if we use autoregressive, we need to use the previous output as the input
        if self.autoregressive:
            prev_outputs = H_t['prev_outputs']
            x = torch.cat((x, prev_outputs), dim=-1)
        
        previous_treatments = H_t['prev_treatments']
        x = torch.cat((x, previous_treatments), dim=-1) # (batch_size, seq_length, input_size)
        return x

    def init_hidden_history(self, H_t):
        x = self.build_H_t(H_t)
        Z_t, _ = self.lstm_history(x) # (batch_size, history_length, hidden_dim)
        treatments = H_t['current_treatments']
        y_hat = self.predict_y_history_net(torch.cat((Z_t, treatments), dim=-1))
        loss = nn.MSELoss()(y_hat, H_t['outputs'])
        return Z_t[:, -1, :], loss
    
    def forward(self, Z_s_prev, a_s, H_t, Y_target):
        """
        Z_s_prev: (batch_size, z_dim)
        a_s: (batch_size, treatment_dim)
        H_t: (batch_size, history_dim)
        Y_target: (batch_size, output_dim)
        """
        # 如果隐状态未初始化，则初始化
        batch_size = Z_s_prev.size(0)
        if self.hidden_state is None:
            self.init_hidden(batch_size)
            
        # 前向传播
        # input = torch.cat([Z_s_prev, a_s, H_t, Y_target], dim=-1).unsqueeze(1)
        # print(f'Z_s_prev shape: {Z_s_prev.shape}, a_s shape: {a_s.shape}, Y_target shape: {Y_target.shape}')
        input = torch.cat([Z_s_prev, Y_target], dim=-1).unsqueeze(1)
        if self.do:
            input = torch.cat([Z_s_prev, Y_target, a_s], dim=-1).unsqueeze(1)

        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            input, (self.hidden_state, self.cell_state)
        )
        lstm_out = lstm_out.squeeze(1)
        lstm_out = torch.cat([lstm_out], dim=-1)

        # input = torch.cat([Z_s_prev, Y_target], dim=-1).unsqueeze(1)
        # lstm_out = self.transition_network(input, a_s)
        
        q_mu = self.fc_mu(lstm_out)
        q_logvar = self.fc_logvar(lstm_out)
        return q_mu, q_logvar
    
    def reset_states(self):
        """重置LSTM隐状态"""
        self.hidden_state = None
        self.cell_state = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterize_multiple(self, mu, logvar, num_samples=10):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, *mu.shape).to(mu.device)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)