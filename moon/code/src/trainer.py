"""PyTorch trainer module.

- Author: Jongkuk Lim, Junghoon Kim
- Contact: lim.jeikei@gmail.com, placidus36@gmail.com
"""

import os
import shutil
from typing import Optional, Tuple, Union
from src.augmentation.dataset import CustomNoisyStudentTrainingDataset
from src.loss import CustomKLLoss
from src.model import Model

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from tqdm import tqdm

from src.utils.torch_utils import save_model
import wandb


def _get_n_data_from_dataloader(dataloader: DataLoader) -> int:
    """Get a number of data in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A number of data in dataloader
    """
    if isinstance(dataloader.sampler, SubsetRandomSampler):
        n_data = len(dataloader.sampler.indices)
    elif isinstance(dataloader.sampler, SequentialSampler):
        n_data = len(dataloader.sampler.data_source)
    else:
        n_data = len(dataloader) * dataloader.batch_size if dataloader.batch_size else 1

    return n_data


def _get_n_batch_from_dataloader(dataloader: DataLoader) -> int:
    """Get a batch number in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A batch number in dataloader
    """
    n_data = _get_n_data_from_dataloader(dataloader)
    n_batch = dataloader.batch_size if dataloader.batch_size else 1

    return n_data // n_batch


def _get_len_label_from_dataset(dataset: Dataset) -> int:
    """Get length of label from dataset.

    Args:
        dataset: torch dataset

    Returns:
        A number of label in set.
    """
    if isinstance(dataset, torchvision.datasets.ImageFolder) or isinstance(
        dataset, torchvision.datasets.vision.VisionDataset
    ):
        return len(dataset.classes)
    elif isinstance(dataset, torch.utils.data.Subset):
        return _get_len_label_from_dataset(dataset.dataset)
    else:
        raise NotImplementedError



class NSTTrainer:
    """Noisy Student Training Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
        model_name:str ="model_name",
        parent_cfg = "",
        parent_weights ="",
        test_dir = "/opt/ml/data/test/NoLabel"
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """

        self.model = model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device

        wandb.init(
            project='p-stgae_model_optimization',
            name= model_name
        )
        if parent_cfg:
            assert parent_weights
            assert os.path.isfile(parent_cfg) and os.path.isfile(parent_weights)
            print('Parent model :')
            self.parent_model = Model(parent_cfg, verbose=True).model
            
            self.parent_model.load_state_dict(
                    torch.load(parent_weights, map_location=device))
            self.parent_model.to(device)
            self.parent_model.eval()

        pseudo_dataset = CustomNoisyStudentTrainingDataset(root=test_dir)
        self.pseudo_dataloader = DataLoader(
            dataset=pseudo_dataset,
            pin_memory=(torch.cuda.is_available()),
            shuffle=True,
            batch_size=32,
            num_workers=4,
            drop_last=True)

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """

        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            self.model.train()

 
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output_teacher = self.parent_model(data)
                else:
                    output_teacher = self.parent_model(data)
                output_teacher = torch.squeeze(output_teacher)
                
                
                loss = self.criterion(outputs, labels,output_teacher)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step(epoch=epoch)

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )
            
                wandb.log({
                    "Train/loss" : running_loss / (batch + 1),
                    "Train/Acc": (correct / total) * 100 ,
                    "Train/F1(macro)" : f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0),
                    'LR' : self.optimizer.param_groups[0]['lr']
                    })

            pbar.close()

            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(self.pseudo_dataloader), total=len(self.pseudo_dataloader))
            for batch, (data, aug_data) in pbar:
                data, aug_data = data.to(self.device), aug_data.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(aug_data)
                else:
                    outputs = self.model(aug_data)
                outputs = torch.squeeze(outputs)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output_teacher = self.parent_model(data)
                else:
                    output_teacher = self.parent_model(data)
                output_teacher = torch.squeeze(output_teacher)
                
                
                loss = self.criterion(outputs, None,output_teacher)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step(epoch=epoch)

                _, pred = torch.max(outputs, 1)
                _, labels = torch.max(output_teacher, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"student/teacher_Train: [{epoch + 1:03d}] "
                    f"student/teacher_Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"student/teacher_Acc: {(correct / total) * 100:.2f}% "
                    f"student/teacher_F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )
            
                wandb.log({
                    "student/teacher_Train/loss" : running_loss / (batch + 1),
                    "student/teacher_Train/Acc": (correct / total) * 100 ,
                    "student/teacher_Train/F1(macro)" : f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0),
                    'LR' : self.optimizer.param_groups[0]['lr']
                    })

            pbar.close()



            test_loss, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            wandb.log({
                "val/loss" : test_loss,
                "val/Acc" : test_acc*100,
                "val/F1(macro)" : test_f1,
                'epoch' : epoch
            })

            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )

        return best_test_acc, best_test_f1

    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels,None).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy



class KLTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
        model_name:str ="model_name",
        parent_cfg = "",
        parent_weights =""
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """

        self.model = model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
        wandb.init(
            project='p-stgae_model_optimization',
            name= model_name
        )
        if parent_cfg:
            assert parent_weights
            assert os.path.isfile(parent_cfg) and os.path.isfile(parent_weights)
            print('Parent model :')
            self.parent_model = Model(parent_cfg, verbose=True).model
            
            self.parent_model.load_state_dict(
                    torch.load(parent_weights, map_location=device))
            self.parent_model.to(device)
            self.parent_model.eval()

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """

        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output_teacher = self.parent_model(data)
                else:
                    output_teacher = self.parent_model(data)
                output_teacher = torch.squeeze(output_teacher)
                
                
                loss = self.criterion(outputs, labels,output_teacher)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step(epoch=epoch)

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )
            
                wandb.log({
                    "Train/loss" : running_loss / (batch + 1),
                    "Train/Acc": (correct / total) * 100 ,
                    "Train/F1(macro)" : f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0),
                    'LR' : self.optimizer.param_groups[0]['lr']
                    })

            pbar.close()

            test_loss, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            wandb.log({
                "val/loss" : test_loss,
                "val/Acc" : test_acc*100,
                "val/F1(macro)" : test_f1
            })

            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )

        return best_test_acc, best_test_f1

    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels,None).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy



class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
        model_name:str ="model_name"
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """

        self.model = model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
        wandb.init(
            project='p-stgae_model_optimization',
            name= model_name
        )

    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """

        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)


                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step(epoch=epoch)

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )
            
                wandb.log({
                    "Train/loss" : running_loss / (batch + 1),
                    "Train/Acc": (correct / total) * 100 ,
                    "Train/F1(macro)" : f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0),
                    'LR' : self.optimizer.param_groups[0]['lr']
                    })

            pbar.close()

            test_loss, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            wandb.log({
                "val/loss" : test_loss,
                "val/Acc" : test_acc*100,
                "val/F1(macro)" : test_f1
            })

            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )

        return best_test_acc, best_test_f1

    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
