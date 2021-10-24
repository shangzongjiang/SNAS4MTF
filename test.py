import logging

logging.basicConfig(level=logging.INFO)

import random
import numpy as np
import torch
from setting import config as cfg
from utils.dataset import TrafficDataset
from model.net import Net
from run_model import RunModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def system_init():
    """ Initialize random seed. """
    random.seed(cfg.sys.seed)
    np.random.seed(cfg.sys.seed)
    torch.manual_seed(cfg.sys.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def main(num_epoch):
    system_init()

    # load data
    dataset = TrafficDataset(
        path=cfg.data.path,
        train_prop=cfg.data.train_prop,
        valid_prop=cfg.data.valid_prop,
        num_nodes=cfg.data.num_nodes,
        in_length=cfg.data.in_length,
        out_length=cfg.data.out_length,
        batch_size_per_gpu=cfg.data.batch_size_per_gpu,
        num_gpus=1,  # torch.cuda.device_count()
        device = device,
    )

    net = Net(
        in_length=cfg.data.in_length,
        out_length=cfg.data.out_length,
        num_nodes = cfg.data.num_nodes,
        node_emb_dim = cfg.model.node_emb_dim,
        graph_hidden = cfg.model.graph_hidden,
        in_channels=cfg.data.in_channels,
        out_channels=cfg.data.out_channels,
        hidden_channels=cfg.model.hidden_channels,
        scale_channels=cfg.model.scale_channels,
        end_channels=cfg.model.end_channels,
        layer_structure=cfg.model.layer_structure,
        num_layer_per_cell=cfg.model.num_layer_per_cell,
        candidate_op_profiles_1=cfg.model.candidate_op_profiles_1,
        candidate_op_profiles_2=cfg.model.candidate_op_profiles_2
    )

    run_model = RunModel(
        name=cfg.model.name,
        net=net,
        dataset=dataset,

        arch_lr=cfg.trainer.arch_lr,
        arch_lr_decay_milestones=cfg.trainer.arch_lr_decay_milestones,
        arch_lr_decay_ratio=cfg.trainer.arch_lr_decay_ratio,
        arch_decay=cfg.trainer.arch_decay,
        arch_clip_gradient=cfg.trainer.arch_clip_gradient,

        weight_lr=cfg.trainer.weight_lr,
        weight_lr_decay_milestones=[20, 40, 60, 80],  # cfg.trainer.weight_lr_decay_milestones,
        weight_lr_decay_ratio=cfg.trainer.weight_lr_decay_ratio,
        weight_decay=cfg.trainer.weight_decay,
        weight_clip_gradient=cfg.trainer.weight_clip_gradient,

        num_search_iterations=cfg.trainer.num_search_iterations,
        num_search_arch_samples=cfg.trainer.num_search_arch_samples,
        num_train_iterations=cfg.trainer.num_train_iterations,

        criterion=cfg.trainer.criterion,
        metric_names=cfg.trainer.metric_names,
        metric_indexes=cfg.trainer.metric_indexes,
        print_frequency=cfg.trainer.print_frequency,

        device_ids=cfg.model.device_ids
    )

    run_model.load(mode='train')
    run_model.clear_records()
    run_model.initialize()
    print('# of params', run_model._net.num_weight_parameters())
    run_model.train(num_epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()

    cfg.load_config(args.config)
    main(args.epoch)
