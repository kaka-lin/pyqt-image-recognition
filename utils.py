import os
import errno

import click
import torch
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False

def load_data():
    #trans = transforms.ToTensor()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = MNIST('./data', download=True, transform=trans)
    test_data = MNIST('./data', train=False, transform=trans)

    return (train_data, test_data)

def metrics_report_func(x):
    if x is not None:
        loss, accuracy = x
        return 'loss: {:.4f} - acc: {:.4f}'.format(loss.item(), accuracy)

def train_model(model, train_loader, optimizer, loss, batch_size):
    model.train()
    _accuracy = 0
    _error = torch.tensor([0])
    with click.progressbar(iterable=train_loader,
                           show_pos=True, show_percent=True,
                           fill_char='#', empty_char=' ',
                           label='train', width=30,
                           item_show_func=metrics_report_func) as bar:
        for batch_idx, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            output = model(data)
            error =  loss(output, target)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / batch_size

            bar.current_item = [error, accuracy]
            _error = error
            _accuracy = accuracy

            optimizer.zero_grad()
            error.backward()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)

            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))

        bar.current_item = [_error, _accuracy]
        bar.render_progress()

def test_model(model, test_loader, loss, batch_size):
    model.eval()
    test_error = 0
    correct = 0
    with click.progressbar(iterable=test_loader,
                           show_pos=True, show_percent=True,
                           fill_char='#', empty_char=' ',
                           label='test', width=30,
                           item_show_func=metrics_report_func) as bar:

        with torch.no_grad():
            for data, target in bar:
                output = model(data)
                test_error +=  loss(output, target)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

                bar.current_item = None

        accuracy = correct / len(test_loader.dataset)
        test_error /= (len(test_loader.dataset)/batch_size)
        bar.current_item = [test_error, accuracy]
        bar.render_progress()

def save_model(model, path='./', mode='train',
               model_name='model',**kwargs):
    if mode == 'checkpoint':
        path = path + 'models/pre_trains/{}_checkpoint.tar'.format(model_name)
    else:
        path = path + 'models/pre_trains/{}.pkl'.format(model_name)

    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if mode == 'inference':
        torch.save(model.state_dict(), path)
    elif mode == 'checkpoint':
        torch.save({
            'model_state_dict': model.state_dict(),
            **kwargs
        }, path)
    else:
        torch.save(model, path)
