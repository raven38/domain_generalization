from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from torch.nn import init
from dataloader import *
from sklearn.decomposition import PCA


def show_images(images, cols = 1, titles = None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        print('model.features parameters are fixed')
        for param in model.parameters():
            param.requires_grad = False


def split_domain(domains, split_idx, print_domain=True):
    source_domain = deepcopy(domains)
    target_domain = [source_domain.pop(split_idx)]
    if print_domain:
        print('Source domain: ', end='')
        for domain in source_domain:
            print(domain, end=', ')
        print('Target domain: ', end='')
        for domain in target_domain:
            print(domain)
    return source_domain, target_domain

domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'PACS_random_split': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ['Caltech', 'Labelme', 'Pascal', 'Sun']
}

def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]

