import os
from pathlib import Path

import click
import numpy as np
import pytorch_pfn_extras as ppe
import torch

from torchvision import models, datasets
from pytorch_pfn_extras import writing
from pytorch_pfn_extras.training import extensions
from torch import nn, optim


def filter_dataset(dataset, f):
    if hasattr(dataset, 'targets'):
        dataset.data, dataset.targets = list(zip(*list(filter(lambda x: f(x[1]), zip(dataset.data, dataset.targets)))))
    elif hasattr(dataset, 'labels'):
        dataset.data, dataset.labels = list(zip(*list(filter(lambda x: f(x[1]), zip(dataset.data, dataset.labels)))))
    else:
        assert False, 'dataset object need to have targets or labels attributes'
    return dataset

def train(gpu, save_path, snapshot, batch_size):
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu) if torch.cuda.is_available() else 'cpu'
    model = models.resnet13(n_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def to_device_optimizer(opt):
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    to_device_optimizer(optimizer)

    train_dataset = datasets.MNIST(root='./data/', train=True)
    test_dataset = datasets.MNIST(root='./data/', train=False)
    train_dataset = filter_dataset(train_dataset, lambda x: x<5)
    test_dataset = filter_dataset(test_dataset, lambda x: x<5)
    print(f'train dataset size: {len(train_dataset)}')
    print(f'test dataset size: {len(test_dataset)}')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            nn.init.zeros_(m.bias)
    model.apply(init_weight)

    iters_per_epoch = len(train_loader)
    if gpu == 0:
        extension = [
            extensions.LogReport(),
            extensions.ProgressBar(),
            extensions.observe_lr(optimizer=optimizer),
            extensions.PlotReport(
                ['loss', 'acc', 'test loss', 'test acc'], 'epoch', filename='loss.png'),
            extensions.PrintReport(['epoch', 'iteration',
                                    'loss', 'acc', 'test loss', 'test acc',
                                    'lr']),
        ]
    else:
        extension = None

    epoch = 100
    manager = ppe.training.ExtensionsManager(
        model, optimizer, epoch,
        extensions=extension,
        iters_per_epoch=iters_per_epoch,
        out_dir=save_path,
        stop_trigger=None)

    #writer = writing.ProcessWriter(savefun=torch.save, out_dir=save_path)
    manager.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    manager.extend(extensions.snapshot(filename='model_{.epoch}', target=model), trigger=(10, 'epoch'))

    if snapshot is not None:
        manager.load_state_dict(torch.load(snapshot))

    criterion = nn.CrossEntropyLoss()
    while not manager.stop_trigger:
        for i, batch in enumerate(train_loader):
            with manager.run_iteration():
                x, t = batch
                x = x.to(device)
                t = t.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, t)
                loss.backward()
                optimizer.step()
                
                accuracy = (output.argmax(dim=1) == target).mean()
                ppe.reporting.report({
                    'loss': loss.item(),
                    'acc': accuracy.item(),
                })
        
        with torhc.no_grad():
            model.eval()
            for i, batch in enumerate(test_loader):
                x, t = batch
                x, t, = x.to(device), t.to(device)
                output = model(x)
                loss = criterion(output, t)

                accuracy = (output.argmax(dim=1) == target).mean()
                ppe.reporting.report({
                    'test loss': loss.item(),
                    'test acc': accuracy.item(),
                })                
            model.train()
@click.command()
@click.option('--save_path', default='checkpoint/test', type=Path)
@click.option('--snapshot', default=None, type=Path)
@click.option('--batch_size', default=8, type=int)
@click.option('--mode', default='train', type=click.Choice(['train', 'eval', 'extract']))
def main(save_path, snapshot, batch_size, mode):
    if mode == 'train':
        train(0, save_path, snapshot, batch_size)


if __name__ == '__main__':
    main()
