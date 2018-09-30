import csv
import json
from tqdm import tqdm
import os
import dill
import urllib
import urllib.request
import PIL
from PIL import Image
import itertools
import numpy as np

import logging

import torch
import torch.nn as nn
import torch.utils.data
import torchvision

import data
from data.openimagedataset import OpenImageDataset

import models
from models import YoloClassifier

log = logging.getLogger(__name__)

def train():
    device = torch.device('cuda')
    #device = torch.device('cpu')

    # Data
    train = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/', output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset', train=True, label_depth=2)
    test = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/', output_dir='/NOBACKUP/hhuang63/oid/OpenImageDatasetValidation', train=False, label_depth=2)
    train.merge_labels(test)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=50, num_workers=20, collate_fn=data.collate, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=50, num_workers=10, collate_fn=data.collate)

    # Check that the labels matchs
    if train.labels_list != test.labels_list:
        raise Exception('Training and test labels mismatch')

    # Init neural net
    net = YoloClassifier(labels=train.labels_list)
    #net.load_state_dict(torch.load('weights/classifier-leaf-7.pt'))
    net = net.to(device)

    # Training loop
    opt = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for iteration in range(1000):
        print('Iteration %d\t (Saving...)' % iteration)
        torch.save(net.state_dict(), 'weights/classifier-pretrain-%d.pt' % iteration)

        total_test_loss = 0
        net.eval()
        for y,x in tqdm(test_dataloader, desc='Testing'):
            x = x.to(device)
            y = y.to(device)

            y_hat = net(x)
            loss = criterion(y_hat, y)
            total_test_loss += loss.item()
        print('Testing Loss: %f' % (total_test_loss/len(test_dataloader)))

        total_loss = 0
        net.train()
        for y,x in tqdm(train_dataloader, desc='Training'):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            y_hat = net(x)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print('Training Loss: %f' % (total_loss/len(train_dataloader)))

        print('%s & %s & %s' % (iteration, (total_loss/len(train_dataloader)), (total_test_loss/len(test_dataloader))))

def show_net_output():
    def sort_predictions(pred):
        """
        Given a size (1,num_classes) tensor consisting of log softmax outputs, return a new array sorted by the log softmax probability.
        Output consists of the original log softmax outputs and the original index.
        """
        return sorted([(i,p) for i,p in enumerate(pred.detach().cpu().numpy()[0,:])], key=lambda x: x[1], reverse=True)

    device = torch.device('cuda')

    # Init neural net
    net = YoloClassifier()
    net.load_state_dict(torch.load('weights/classifier-baked-1.pt'))
    net = net.to(device)

    for j in range(5):
        y = torch.nn.functional.softmax(net(test[j][1].view(1,3,224,224).to(device)))
        sorted_y = sort_predictions(y)
        print("Actual class: %s" % data.get_class_description(test[j][0]))
        for i in range(5):
            pred = data.get_class_description(sorted_y[i][0])
            percent = sorted_y[i][1]
            print("\t%s - %f" % (pred, percent))

def predict(weights_file_name, img_file_name):
    device = torch.device('cpu')

    # Data (Load this to get the labels)
    train = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/', output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset', train=True, label_depth=2)
    test = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/', output_dir='/NOBACKUP/hhuang63/oid/OpenImageDatasetValidation', train=False, label_depth=2)
    train.merge_labels(test)
    labels = train.labels_list

    net = YoloClassifier(labels=labels)
    net.load_state_dict(torch.load(weights_file_name))
    net = net.to(device)
    net.eval()

    # Load image
    img = Image.open(img_file_name)
    # Process image
    min_dim = min(img.size)
    max_dim = max(img.size)
    scale = 225/min_dim
    img.thumbnail([max_dim*scale,max_dim*scale])
    left = img.size[0]//2-112
    top = img.size[1]//2-112
    img = img.crop([left,top,left+224,top+224])
    img = img.convert('RGB')
    transform = torchvision.transforms.ToTensor()
    img = transform(img)
    # Feed image to neural net
    output = net(img.view(-1,3,224,224))
    output = output.argmax().item()
    # Human-readable output
    class_descriptions = ClassDescriptions(input_dir=input_dir,output_dir=output_dir)
    class_descriptions.load()
    description = class_description[labels[output]]
    # Output/return prediction
    print(description)
    return description

if __name__ == "__main__":
    input_dir = '/NOBACKUP/hhuang63/oid/'
    output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset'
    #train()
    #predict('weights/classifier-leaf-28.pt','875806_R.jpg')

    predict(weights_file_name, img_file_name)
