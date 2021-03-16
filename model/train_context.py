from os.path import join, exists

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class TrainContext:
    gru_network: nn.Module
    seg_network: nn.Module
    flow_network: nn.Module

    gru_optimizer: Optimizer
    seg_optimizer: Optimizer

    def __init__(self, seg_network, flow_network, gru_network, seg_optimizer, gru_optimizer):
        self.seg_network = seg_network
        self.flow_network = flow_network
        self.gru_network = gru_network
        self.gru_optimizer = gru_optimizer
        self.seg_optimizer = seg_optimizer

        self.max_iou = 0
        self.last_completed_epoch = 0
        self.epoch = 0

    def save_checkpoint(self, filename):
        data = {
            'seg_network':          self.seg_network.state_dict(),
            'flow_network':         self.flow_network.state_dict(),
            'gru_network':          self.gru_network.state_dict(),
            'seg_optimizer':        self.seg_optimizer.state_dict(),
            'gru_optimizer':        self.gru_optimizer.state_dict(),
            'best_acc':             self.max_iou,
            'last_completed_epoch': self.last_completed_epoch,
        }

        torch.save(data, filename)

    def load_checkpoint(self, filename):
        data = torch.load(filename)

        self.seg_network.load_state_dict(data['seg_network'], strict=True)
        self.flow_network.load_state_dict(data['flow_network'], strict=True)
        self.gru_network.load_state_dict(data['gru_network'], strict=True)
        self.seg_optimizer.load_state_dict(data['seg_optimizer'])
        self.gru_optimizer.load_state_dict(data['gru_optimizer'])

        self.max_iou = data['best_acc']
        self.last_completed_epoch = data['last_completed_epoch']

    def save_net_state_dicts(self, base_filename, seg_network=True, flow_network=True, gru_network=True):
        if seg_network:
            torch.save(self.seg_network.state_dict(), base_filename + '_seg.pth')
        if flow_network:
            torch.save(self.flow_network.state_dict(), base_filename + '_flow.pth')
        if gru_network:
            torch.save(self.gru_network.state_dict(), base_filename + '_gru.pth')

    def load_net_state_dicts(self, base_directory, weights_filename, seg_network=True, flow_network=True, gru_network=True):
        assert exists(base_directory), f'{base_directory} not existent'
        base_filename = join(base_directory, weights_filename)
        if seg_network:
            self.seg_network.load_state_dict(torch.load(base_filename + '_seg.pth'), strict=True)
        if flow_network:
            self.flow_network.load_state_dict(torch.load(base_filename + '_flow.pth'), strict=True)
        if gru_network:
            self.gru_network.load_state_dict(torch.load(base_filename + '_gru.pth'), strict=True)
