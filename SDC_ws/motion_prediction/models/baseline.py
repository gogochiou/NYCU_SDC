import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from models.subnets.subnets import MultiheadAttention, MLP, MapNet, SubGraph
import yaml 

import math

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
_output_heads = int(config['net']['output_heads'])

''' Model
'''
class Baseline(pl.LightningModule):
    def __init__(self):
        super(Baseline, self).__init__()
        ''' history state (x, y, vx, vy, yaw, object_type) * 5s * 10Hz
        '''
        self.history_encoder = MLP(6*5*10, 128, 128) ## state*5s*10Hz
        self.lane_encoder = MapNet(2, 128, 128, 10)
        self.neighbor_encoder = MapNet(6, 128, 128, 11)
        self.lane_attn = MultiheadAttention(128, 8)
        self.neighbor_attn = MultiheadAttention(128, 8)
        
        trajs = []
        confs = []
        ''' we predict 6 different future trajectories to handle different possible cases.
        '''
        for i in range(6):
            ''' future state (x, y, vx, vy, yaw) * 6s * 10Hz
            '''
            trajs.append(
                MLP(128, 256, 5*6*10) ## state*6s*10Hz
                )
            ''' we use model to predict the confidence score of prediction
            '''
            confs.append(
                    nn.Sequential(
                    MLP(128, 64, 1),
                    nn.Sigmoid()
                    )
                )
        self.future_decoder_traj = nn.ModuleList(trajs)
        self.future_decoder_conf = nn.ModuleList(confs)

    def forward(self, data):
        ''' In deep learning, data['x'] means input, data['y'] means groundtruth
        '''
        x = data['x'].reshape(-1, 6*5*10) ## 6 state
        x = self.history_encoder(x)
        	
        lane = data['lane_graph']
        lane = self.lane_encoder(lane)

        neighbor = data['neighbor_graph'].reshape(-1, 11, 6)
        neighbor = self.neighbor_encoder(neighbor)
        
        x = x.unsqueeze(0)
        lane = lane.unsqueeze(0)
        neighbor = neighbor.unsqueeze(0)

        lane_mask = data['lane_mask']
        lane_attn_out = self.lane_attn(x, lane, lane, attn_mask=lane_mask) 
        
        x = x + lane_attn_out

        neighbor_mask = data['neighbor_mask']
        neighbor_attn_out = self.neighbor_attn(x, neighbor, neighbor, attn_mask=neighbor_mask)

        x = x + neighbor_attn_out
        x = x.squeeze(0)
        
        trajs = []
        confs = []
        for i in range(6):
            trajs.append(self.future_decoder_traj[i](x))
            confs.append(self.future_decoder_conf[i](x))
        trajs = torch.stack(trajs, 1)
        confs = torch.stack(confs, 1)
        
        return trajs, confs
	




