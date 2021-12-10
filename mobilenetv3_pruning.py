# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for supported l1norm and l2norm pruning algorithms.
In this example, we show the end-to-end pruning process: pre-training -> pruning -> fine-tuning.
Note that pruners use masks to simulate the real pruning. In order to obtain a real compressed model, model speed up is required.

'''
import argparse

import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L1NormPruner, L2NormPruner
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
g_epoch = 0


def trainer(model, optimizer, criterion):
    global g_epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx and batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                g_epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    g_epoch += 1

def evaluator(model):
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    print('Accuracy: {}%\n'.format(acc))
    return acc

def optimizer_scheduler_generator(model, _lr=0.1, _momentum=0.9, _weight_decay=5e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr=_lr, momentum=_momentum, weight_decay=_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(args.pretrain_epochs * 0.5), int(args.pretrain_epochs * 0.75)], gamma=0.1)
    return optimizer, scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/mobilenetv3.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    parser.add_argument('--pruner', type=str, default='l1norm',
                        choices=['l1norm', 'l2norm'],
                        help='pruner to use')
    parser.add_argument('--pretrain-epochs', type=int, default=1,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--fine-tune-epochs', type=int, default=1,
                        help='number of epochs to fine tune the model')
    args = parser.parse_args()

    print('\n' + '=' * 50 + ' START TO TRAIN THE MODEL ' + '=' * 50)
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))

    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    # print(f"Model save path: {model_path}")
    # # if os.path.isfile(model_path):
    # #     model_instance.model.load_state_dict(
    # #         torch.load(model_path, map_location=device)
    # #     )
    # model_instance.model.to(device)
    model=model_instance.model
    model = mobilenetv3_large()
    model.to(device)
    # print(model)
    flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))

#     model.load_state_dict(torch.load(model_path))
# #     # Create dataloader
    train_loader, val_dl, test_loader = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_loader),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler()
    )
    pre_best_acc = -0.1
    best_state_dict = None
    sparsity = 0.4
    for i in range(1):
        trainer(model, optimizer, criterion)
        scheduler.step()
        acc = evaluator(model)
        if acc > pre_best_acc:
            pre_best_acc = acc
            best_state_dict = model.state_dict()
    print("Best accuracy: {}".format(pre_best_acc))
    model.load_state_dict(best_state_dict)
    pre_flops, pre_params, _ = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))
    g_epoch = 0

    # Start to prune and speedup
    print('\n' + '=' * 50 + ' START TO PRUNE THE BEST ACCURACY PRETRAINED MODEL ' + '=' * 50)
    config_list = [{
            'op_names': ['features.{}.conv.0'.format(x) for x in range(2, 16)],
            'sparsity': sparsity
        },{
            'op_names': ['features.{}.conv.3'.format(x) for x in range(2, 16)],
            'sparsity': sparsity
        },
{
            'op_names': ['features.{}.conv.7'.format(x) for x in range(2, 16)],
            'sparsity': sparsity
        }]
    print(config_list)
    if 'l1' in args.pruner:
        pruner = L1NormPruner(model, config_list)
    else:
        pruner = L2NormPruner(model, config_list)
    _, masks = pruner.compress()
    pruner.show_pruned_weights()
    pruner._unwrap_model()
    ModelSpeedup(model, dummy_input=torch.rand([128, 3, 32, 32]).to(device), masks_file=masks).speedup_model()
    print('\n' + '=' * 50 + ' EVALUATE THE MODEL AFTER SPEEDUP ' + '=' * 50)
    evaluator(model)

    # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
    print('\n' + '=' * 50 + ' START TO FINE TUNE THE MODEL ' + '=' * 50)
    optimizer, scheduler = optimizer_scheduler_generator(model, _lr=0.01)

    best_acc = 0.0
    for i in range(1):
        trainer(model, optimizer, criterion)
        scheduler.step()
        best_acc = max(evaluator(model), best_acc)
    flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))
    print(f'Pretrained model FLOPs {pre_flops/1e6:.2f} M, #Params: {pre_params/1e6:.2f}M, Accuracy: {pre_best_acc: .2f}%')
    print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_acc: .2f}%')
