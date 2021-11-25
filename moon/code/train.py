"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.dataloader import create_dataloader
from src.loss import CustomCriterion, CustomKLLoss
from src.model import Model
from src.trainer import KLTrainer, NSTTrainer, TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info

from src.modules.swin import set_weight_decay
from timm.scheduler.cosine_lr import CosineLRScheduler

def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
    model_name : str,
    parent_cfg : str,
    parent_weights : str,
    noisy_train:bool,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion

    ## custom
    EPS = 1e-8
    BETAS = (0.9, 0.999)
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.05
    MIN_LR = data_config["INIT_LR"]/100
    WARMUP_LR = MIN_LR/10
    WARMUP_EPOCHS = 10
    
    if 'swin' in model_name.split('_')[0]:
        skip = {}
        skip_keywords = {}
        if hasattr(model_instance.model, 'no_weight_decay'):
            skip = model_instance.model.no_weight_decay()
        if hasattr(model_instance.model, 'no_weight_decay_keywords'):
            skip_keywords = model_instance.model.no_weight_decay_keywords()
        parameters = set_weight_decay(model_instance.model, skip, skip_keywords)
    else:
        parameters = model_instance.model.parameters()
        
    # optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
    optimizer = optim.AdamW(parameters, eps=EPS, betas=BETAS,lr=data_config["INIT_LR"], weight_decay=WEIGHT_DECAY)
    scheduler = CosineLRScheduler(
        optimizer, 
        t_initial=data_config['EPOCHS'], 
        warmup_t=WARMUP_EPOCHS, 
        warmup_lr_init=WARMUP_LR,
        lr_min = MIN_LR)

    if parent_cfg:
        criterion = CustomKLLoss(device=device, alpha=0.5,T=1.5)
    else:
        criterion = CustomCriterion(
            samples_per_cls=get_label_counts(data_config["DATA_PATH"])
            if data_config["DATASET"] == "TACO"
            else None,
            device=device,
        )
        

    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    if parent_cfg:
        if noisy_train:
            print('>>>> Setting Noisy Training..')
            trainer = NSTTrainer(model_instance.model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                model_path=model_path,
                verbose=1,
                model_name=model_name,
                parent_cfg=parent_cfg,
                parent_weights=parent_weights)
        else:
            print('>>>> Setting knowledge distillation..')
            trainer = KLTrainer(model_instance.model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                model_path=model_path,
                verbose=1,
                model_name=model_name,
                parent_cfg=parent_cfg,
                parent_weights=parent_weights)
    else:
        trainer = TorchTrainer(
            model=model_instance.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            model_path=model_path,
            verbose=1,
            model_name=model_name)

    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/mobilenetv3.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument("--data", default="configs/data/taco.yaml", type=str, help="data config")
    parser.add_argument("--model_name", default="model_name", type=str, help="model name that is shown in wandb")
    parser.add_argument("--parent_cfg", default="", type=str)
    parser.add_argument("--parent_weights", default="", type=str)
    parser.add_argument('--noisy_train',dest='noisy_train',action='store_true')
    parser.set_defaults(noisy_train=False)
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", args.model_name))

    # if os.path.exists(log_dir): 
    #     modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
    #     new_log_dir = os.path.join(log_dir, modified.strftime("%Y-%m-%d_%H-%M-%S"))
    #     os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
        model_name=args.model_name,
        parent_cfg=args.parent_cfg,
        parent_weights = args.parent_weights,
        noisy_train=args.noisy_train
        )

