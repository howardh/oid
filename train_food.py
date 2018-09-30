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
from data.classdescriptions import ClassDescriptions
from data.labelshierarchy import LabelsHierarchy

import models
from models import YoloClassifier

def train():
    device = torch.device('cuda')
    #device = torch.device('cpu')

    # Data
    train = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/',
            output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset', train=True,
            label_depth=2, label_root='Fruit')
    test = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/',
            output_dir='/NOBACKUP/hhuang63/oid/OpenImageDatasetValidation',
            train=False, label_depth=2, label_root='Fruit')
    train.merge_labels(test)
    labels = train.labels_list
    with open('labels-food.pkl','wb') as f:
        dill.dump(labels,f)

    weights_file_name = 'weights/classifier-leaf-28.pt'
    net = YoloClassifier(labels=labels)
    state_dict = net.state_dict()
    trained_state_dict = torch.load(weights_file_name)
    trained_state_dict['linear.bias'] = state_dict['linear.bias']
    trained_state_dict['linear.weight'] = state_dict['linear.weight']
    net.load_state_dict(trained_state_dict)
    net = net.to(device)
    net.train()

    # Freeze all layers but the last
    for p in itertools.chain(net.layer1.parameters(), net.layer2.parameters(),
            net.layer3.parameters(), net.layer4.parameters(),
            net.layer5.parameters()):
        p.requires_grad = False

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=50,
            num_workers=20, collate_fn=data.openimagedataset.collate, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=50,
            num_workers=10, collate_fn=data.openimagedataset.collate)

    opt = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for iteration in range(1000):
        print('Iteration %d\t (Saving...)' % iteration)
        torch.save(net.state_dict(), 'weights/classifier-fruit-%d.pt' % iteration)

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

def predict(weights_file_name, labels_file_name, img_file_name):
    device = torch.device('cpu')

    # Data (Load this to get the labels)
    try:
        with open('labels/labels-food.pkl','rb') as f:
            labels = dill.load(f)
    except:
        train = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/',
                output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset', train=True,
                label_depth=2, label_root='Fruit')
        test = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/',
                output_dir='/NOBACKUP/hhuang63/oid/OpenImageDatasetValidation',
                train=False, label_depth=2, label_root='Fruit')
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
    description = class_descriptions[labels[output]]
    # Output/return prediction
    print(description)
    return description

if __name__ == "__main__":
    input_dir = '/NOBACKUP/hhuang63/oid/'
    output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset'
    train()
    #predict('weights/classifier-fruit-11.pt','875806_R.jpg')

