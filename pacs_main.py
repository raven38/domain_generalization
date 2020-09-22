import matplotlib
matplotlib.use('Agg')
import os
from pathlib import Path
from itertools import product
import click
import numpy as np
import pytorch_pfn_extras as ppe
import torch
import copy

from torchvision import models, datasets, transforms
from pytorch_pfn_extras import writing
from pytorch_pfn_extras.training import extensions
from torch import nn, optim
import torch.nn.functional as F

from dataloader import random_split_dataloader
from util import get_domain, split_domain


def test(model, data, target, device, criterion):
    with torch.no_grad():
        model.eval()
        x, t, = data.to(device), target.to(device)
        output = model(x)
        loss = criterion(output, t)

        accuracy = (output.argmax(dim=1) == t).float().mean()
        ppe.reporting.report({
            'val/loss': loss.item(),
            'val/acc': accuracy.item(),
        })                
        model.train()


def extract_feature(gpu, save_path, data_root, data, exp_num, snapshot, batch_size):
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu) if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(num_classes=6)
    model.load_state_dict(torch.load(snapshot))
    model = nn.Sequential(*list(model.children())[:-1]).to(device)
    model.eval()

    domain = get_domain(data)
    source_domain, target_domain = split_domain(domain, exp_num)
    source_train, source_val, target_test = random_split_dataloader(
        data=data, data_root=data_root, source_domain=source_domain, target_domain=target_domain,
        batch_size=batch_size, num_workers=4, classes=[0, 1, 2, 3, 4, 5])
    source_train_unknown, source_val_unknown, target_test_unknown = random_split_dataloader(
        data=data, data_root=data_root, source_domain=source_domain, target_domain=target_domain,
        batch_size=batch_size, num_workers=4, classes=[6])

    loaders = [source_train, target_test, source_train_unknown, target_test_unknown]
    names = ['source_known', 'target_known', 'source_unknown', 'target_unknown']

    with torch.no_grad():
        for loader, name in zip(loaders, names):
            features = []
            labels = []
            for x, t in loader:
                x = x.to(device)
                output = model(x)
                output = torch.flatten(output, start_dim=1)
                features.append(output.cpu().numpy())
                labels.append(t.cpu().numpy())
            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)
            np.savez(f'{str(save_path)}/{name}.npz', features=features, labels=labels)


def train(gpu, save_path, data_root, data, exp_num, snapshot, batch_size):
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu) if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(num_classes=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def to_device_optimizer(opt):
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    to_device_optimizer(optimizer)

    domain = get_domain(data)
    source_domain, target_domain = split_domain(domain, exp_num)

    source_train, source_val, target_test = random_split_dataloader(
        data=data, data_root=data_root, source_domain=source_domain, target_domain=target_domain,
        batch_size=batch_size, num_workers=4, classes=[0, 1, 2, 3, 4, 5])

    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            nn.init.zeros_(m.bias)
    model.apply(init_weight)
    criterion = nn.CrossEntropyLoss()
    iters_per_epoch = len(source_train)
    if gpu == 0:
        extension = [
            extensions.LogReport(),
            extensions.ProgressBar(),
            extensions.observe_lr(optimizer=optimizer),
            extensions.Evaluator(
                source_val, model,
                eval_func=lambda data, target: test(model, data, target, device, criterion),
                progress_bar=False),
            extensions.PlotReport(
                ['loss', 'acc', 'val/loss', 'val/acc'], 'epoch', filename='loss.png'),
            extensions.PrintReport(['epoch', 'iteration',
                                    'loss', 'acc', 'val/loss', 'val/acc',
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

    manager.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    manager.extend(extensions.snapshot(filename='model_{.epoch}', target=model), trigger=(10, 'epoch'))

    if snapshot is not None:
        manager.load_state_dict(torch.load(snapshot))

    while not manager.stop_trigger:
        for i, batch in enumerate(source_train):
            with manager.run_iteration():
                x, t = batch
                x = x.to(device)
                t = t.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, t)
                loss.backward()
                optimizer.step()
                
                accuracy = (output.argmax(dim=1) == t).float().mean()
                ppe.reporting.report({
                    'loss': loss.item(),
                    'acc': accuracy.item(),
                })


def eval(gpu, save_path, data_root, data, exp_num, threshold, snapshot, batch_size):
    torch.cuda.set_device(gpu)
    device = torch.device('cuda', gpu) if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(num_classes=6)
    model.load_state_dict(torch.load(snapshot))
    model, head = nn.Sequential(*list(model.children())[:-1]), nn.Sequential(*list(model.children())[-1:])
    model, head = model.to(device), head.to(device)
    model.eval()
    head.eval()

    domain = get_domain(data)
    source_domain, target_domain = split_domain(domain, exp_num)
    source_train, _, _ = random_split_dataloader(
        data=data, data_root=data_root, source_domain=source_domain, target_domain=target_domain,
        batch_size=batch_size, num_workers=4, classes=[0, 1, 2, 3, 4, 5])
    _, source_eval, target_eval = random_split_dataloader(
        data=data, data_root=data_root, source_domain=source_domain, target_domain=target_domain,
        batch_size=batch_size, num_workers=4, classes=[0, 1, 2, 3, 4, 5, 6])

    with torch.no_grad():
        sample_mean, sample_precision = estimate_sample_statistics(model, source_train)

    # features = []
    # outputs = []
    # ys = []
    # for x, t in source_train:
    #     x = x.to(device)
    #     t = t.to(device)
    #     feature = torch.flatten(model(x), 1)
    #     output = head(feature)
    #     features.append(feature.cpu().numpy())
    #     outputs.append(output.cpu().numpy())
    #     ys.append(t.cpu().numpy())
    # features = np.concatenate(features, axis=0)
    # ys = np.concatenate(ys, axis=0)
    # mu_hat = np.mean(features, axis=0)
    # sigma_hat = (features - mu_hat).T @ (features - mu_hat) / len(features)
    # np.savez(f'{str(save_path).split("/")[-1]}_mahalanobis_statistics.npz', mu_hat=mu_hat, sigma_hat=sigma_hat)


    # inv_s_hat = np.linalg.inv(sigma_hat)
    features = []
    ys = []
    outputs = []
    domains = []
    mahalanobis_socres = []
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for loader, d in zip([source_eval, target_eval], [np.zeros, np.ones]):
        for x, t in loader:
            x = x.to(device)
            with torch.no_grad():
                feature = torch.flatten(model(x), 1)
                output = head(feature)
            m_score = calc_mahalanobis_score(model, x, sample_mean, sample_precision, m_list)
            features.append(feature.cpu().numpy())
            outputs.append(output.cpu().numpy())
            ys.append(t.numpy())
            domains.append(d(len(feature)))
            mahalanobis_socres.append(m_score.cpu().numpy())
    features = np.concatenate(features, axis=0)
    ys = np.concatenate(ys, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    domains = np.concatenate(domains, axis=0)
    mahalanobis_socres = np.concatenate(mahalanobis_socres, axis=0)
    np.savez(f'{str(save_path).split("/")[-1]}_predicts.npz', feature=features, y=ys, domain=domains, output=outputs, mahalanobis=mahalanobis_socres)

    # entropy = -torch.sum(F.softmax(output, 1) * F.log_softmax(output, 1), 1)
    # unknown = ((entropy > 0.01).float() * (output.max() + 1e-8)).reshape(-1, 1)
    # md = - np.diag((features - mu_hat) @ inv_s_hat @ (features - mu_hat).T)
    # unknown = ((md > threshold).astype(float) * (outputs.max() + 1e-8)).reshape(-1, 1)
    # outputs = np.concatenate([outputs, unknown], axis=1)
    # corrects = (outputs.argmax(axis=1) == ys).astype(float)
    # print(f'acc {corrects.mean()}')


def estimate_sample_statistics(model, train_loader):
    import sklearn.covariance
    device = next(model.parameters()).device
    model.eval()
    glasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    features = []
    for data, _ in train_loader:
        data = data.to(device)
        feature = torch.flatten(model(data), 1)
        features.append(feature)
        
    features = torch.cat(features, dim=0)
    sample_mean = torch.mean(features, dim=0)
    X = features - sample_mean
    glasso.fit(X.cpu().numpy())
    sample_precision = torch.from_numpy(glasso.get_precision()).float().to(device)
    return sample_mean, sample_precision


def calc_mahalanobis_score(model, data, sample_mean, sample_precision, m_list):
    model.eval()

    feature = torch.flatten(model(data), 1)

    zero_f = feature - sample_mean
    m = -0.5 * torch.mm(torch.mm(zero_f, sample_precision), zero_f.t()).diag()
    loss = torch.mean(-m)
    loss.backward()

    def calc_score(magnitude):
        preprocessed_inputs = torch.add(data.detach(), -magnitude, gradient)
        noise_feature = torch.flatten(model(preprocessed_inputs), 1)
        zero_f = noise_feature - sample_mean
        m_score = -0.5 * torch.mm(torch.mm(zero_f, sample_precision), zero_f.t()).diag()
        return m_score

    gradient = torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    with torch.no_grad():
        m_scores = torch.cat([calc_score(m) for m in m_list], dim=1)
    return m_scores


@click.command()
@click.option('--save-path', default='checkpoint/test', type=Path)
@click.option('--data-root', default='/data/', type=Path)
@click.option('--data', default='PACS')
@click.option('--exp-num', default=0)
@click.option('--snapshot', default=None, type=Path)
@click.option('--batch-size', default=32, type=int)
@click.option('--threshold', default=0.1)
@click.option('--mode', default='train', type=click.Choice(['train', 'eval', 'extract']))
def main(save_path, data_root, data, exp_num, threshold, snapshot, batch_size, mode):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path / 'models', exist_ok=True)

    if mode == 'train':
        train(0, save_path, data_root, data, exp_num, snapshot, batch_size)
    elif mode == 'extract':
        extract_feature(0, data_root, data, exp_num, save_path, snapshot, batch_size)
    elif mode == 'eval':
        eval(0, save_path, data_root, data, exp_num, threshold, snapshot, batch_size)

if __name__ == '__main__':
    main()
