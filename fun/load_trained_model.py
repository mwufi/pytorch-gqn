""""""

"""
-------------------   Fun Part 1: Testing your checkpoint ---------------------

Awesome! You've got a trained model, sitting in the `/checkpoints` directory.
Now we want to start doing stuff with it! First, let's make sure we can load
it! That's what this is for

Summary:

> python fun/load_trained_model.py --data=data/dummy

will
1) show a nice picture
2) save the result

"""
import sys
from argparse import ArgumentParser
sys.path.append('.')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from building_blocks.gqn import GenerativeQueryNetwork
from building_blocks.training import partition
from data.datasets import ShepardMetzler, Scene

import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
device = 'cuda:0' if cuda else 'cpu'
map_location = None if cuda else 'cpu'


def load(checkpoint):
    # the fun thing is, since this is a shared-core model, we can change L to be different
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
    weights = torch.load(checkpoint, map_location=map_location)

    # GPU weights have "module." prefixed to everything
    if device == 'cpu':
        cpuWeights = {}
        for k,v in weights.items():
            x = k.replace('module.','')
            cpuWeights[x] = v
        del weights
        weights = cpuWeights

    model.load_state_dict(weights)
    return model


def getLoader(data_dir, batch_size):
    test_dataset = ShepardMetzler(root_dir=data_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    i = iter(test_loader)
    return i


if __name__ == '__main__':
    parser = ArgumentParser(description='Load a checkpoint')
    parser.add_argument('--checkpoint', type=int, help='where is it?', default=20000)
    parser.add_argument('--data', type=str, help='root_dir of the data', default='data/shepard_metzler_5_parts-torch')
    parser.add_argument('--batch_size', type=int, help='number of examples to try', default=4)
    args = parser.parse_args()

    checkpoint = 'checkpoints/checkpoint_model_{}.pth'.format(args.checkpoint)
    model = load(checkpoint)
    model.eval()

    d = getLoader(args.data, args.batch_size)
    with torch.no_grad():
        x, v = next(d)
        x, v = x.to(device), v.to(device)
        x, v, x_q, v_q = partition(x, v)

        # Reconstruction, representation and divergence
        x_mu, _, kl = model(x, v, x_q, v_q)
        x_mu = torch.cat((x_q, x_mu), dim=0)
        print(x_mu.size())

        k = make_grid(x_mu)
        plt.imshow(k.transpose(0,2).transpose(0,1))
        plt.savefig('results.png')
        plt.show()
