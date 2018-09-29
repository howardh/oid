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
from data.classdescriptions import ClassDescriptions
from data.labelshierarchy import LabelsHierarchy
from data.bbox import BoundingBoxes
from data.urls import ImageUrls

log = logging.getLogger(__name__)

class OpenImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir='.', output_dir='OpenImageDataset', label_depth=-1, train=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.original_dir = os.path.join(output_dir, 'original')
        self.resized_dir = os.path.join(output_dir, 'resized')

        self.resized_images = {}

        self.train = train

        log.info('Creating OpenImageDataset directories')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.resized_dir, exist_ok=True)

        log.info('Loading class description mappings')
        class_descriptions = ClassDescriptions(input_dir=input_dir,output_dir=output_dir)
        class_descriptions.load()

        log.info('Loading labels tree')
        labels_hierarchy = LabelsHierarchy(input_dir=input_dir,output_dir=output_dir)
        labels_hierarchy.load()
        if label_depth == -1:
            label_mapping = labels_hierarchy.tree.get_map_to_level(None)
        else:
            label_mapping = labels_hierarchy.tree.get_map_to_level(label_depth)

        log.info('Loading bounding boxes')
        if self.train:
            bounding_boxes = BoundingBoxes(input_dir=input_dir,output_dir=output_dir)
        else:
            bounding_boxes = BoundingBoxes(input_dir=input_dir,output_dir=output_dir,file_name='validation-annotations-bbox.csv')
        bounding_boxes.load()

        log.info('Loading image urls')
        image_urls = ImageUrls(input_dir=input_dir,output_dir=output_dir)
        image_urls.load()

        log.info('Loading other stuff')
        food_labels = labels_hierarchy[class_descriptions['Food']]
        label_map = food_labels.get_map_to_level()
        food_boxes = bounding_boxes[food_labels]

        # TODO: Map from image id to a list of bounding boxes for that image
        # Name images to include bounding box coordinates. This is unambiguous, and no ordering is required.

        dataset = []
        labels = set()
        for row in tqdm(bounding_boxes[food_labels], desc='Creating Dataset'):
            img_id = row[0]
            # label
            label = row[2]
            if label in label_mapping:
                label = label_mapping[label]
            else:
                continue # Skip if there's no label mapping available. e.g., if partial paths are ignored when creating the mapping, and some labels are left out.
            labels.add(label)
            # bounding ox
            xmin = float(row[4])
            xmax = float(row[5])
            ymin = float(row[6])
            ymax = float(row[7])
            # output file name
            original_file_name = os.path.join(self.original_dir, img_id)
            resized_file_name = '%s-(%.2f-%.2f-%.2f-%.2f)' % (img_id, xmin,xmax,ymin,ymax)
            resized_file_name = os.path.join(self.resized_dir, resized_file_name)
            # Save data
            dataset.append({
                'url': image_urls[img_id][0],
                'label': label,
                'bounding_box': {
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax
                },
                'original_file_name': original_file_name,
                'resized_file_name': resized_file_name
            })
        self.dataset = dataset
        self.labels_list = sorted(list(labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.train:
            return self.get_augmented_data(idx)
        else:
            return self.get_testing_data(idx)

    def merge_labels(self, dataset):
        """ Given another OpenImageDataset, merge the labels from both so that both datasets have the same labels and label ordering. """
        labels = self.labels_list + dataset.labels_list
        labels = list(set(labels))
        labels = sorted(labels)
        self.labels_list = labels
        dataset.labels_list = labels

    def get_num_labels(self):
        return len(self.labels_list)

    def get_testing_data(self, idx):
        try:
            data = self.dataset[idx]
            label = torch.Tensor([self.labels_list.index(data['label'])]).long().squeeze()
            resized_file_name = data['resized_file_name']
            original_file_name = data['original_file_name']
            url = data['url']
            transform = torchvision.transforms.ToTensor()
            # Check if resized image exists
            if os.path.isfile(resized_file_name):
                try:
                    img = Image.open(resized_file_name)
                except OSError:
                    log.warning('Error loading file %s. Rebuilding.' % resized_file_name)
                    os.remove(resized_file_name)
            if not os.path.isfile(resized_file_name):
                # Check if original image is saved
                # Warning: Can cause race condition when multithreading and two threads are downloading the same file
                if not os.path.isfile(original_file_name):
                    urllib.request.urlretrieve(url, original_file_name) 
                try:
                    img = Image.open(original_file_name)
                except OSError:
                    #log.warning('Error loading file %s. Redownloading.' % original_file_name)
                    #urllib.request.urlretrieve(url, original_file_name) 
                    #img = Image.open(original_file_name)
                    log.debug('Error loading file %s. Skipping.' % original_file_name)
                    return None

                # Get object bounding box
                xmin = data['bounding_box']['xmin']
                xmax = data['bounding_box']['xmax']
                ymin = data['bounding_box']['ymin']
                ymax = data['bounding_box']['ymax']
                # Target dimensions
                target_size = 224
                def compute_scaling(size, target_size, dim_min, dim_max):
                    box_size = (dim_max-dim_min)*size
                    if target_size > box_size:
                        return 1
                    return target_size/box_size
                xscale = compute_scaling(img.size[0], target_size, xmin, xmax)
                yscale = compute_scaling(img.size[1], target_size, ymin, ymax)
                scale = min(xscale, yscale)
                if min(img.size)*scale < target_size:
                    scale = target_size/min(img.size)
                # resize
                # Resizing to certain sizes for some images don't work? Says image is truncated.
                for correction in range(3):
                    try:
                        img = img.resize([int(scale*img.size[0])+correction, int(scale*img.size[1])+correction])
                        break
                    except OSError:
                        pass
                else:
                    raise
                # Cropping window
                xcentre = (xmin+xmax)/2
                ycentre = (ymin+ymax)/2
                box_left   = int(img.size[0]*xcentre-224/2)
                box_right  = box_left+224
                box_top    = int(img.size[1]*ycentre-224/2)
                box_bottom = box_top+224
                if box_left < 0:
                    box_right -= box_left
                    box_left -= box_left
                elif box_right > img.size[0]:
                    box_left += img.size[0]-box_right
                    box_right += img.size[0]-box_right
                if box_top < 0:
                    box_bottom -= box_top
                    box_top -= box_top
                elif box_bottom > img.size[1]:
                    box_top += img.size[1]-box_bottom
                    box_bottom += img.size[1]-box_bottom
                # crop
                img = img.crop([box_left, box_top, box_right, box_bottom])
                if box_right-box_left != 224 or box_bottom-box_top != 224:
                    print(box_left, box_right, box_top, box_bottom)
                    print(img.size)
                img.save(resized_file_name, format='jpeg')
            # Remove channel if alpha
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                print('Found transparency. Processing alpha channels.')
                alpha = img.split()[3]
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=alpha)
                img = background
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            img = transform(img)
            # Add channels if greyscale
            if img.size()[0] == 1:
                img = img.repeat(3,1,1)
            if img.size()[0] == 4:
                print('Alpha channel found, but not processed: %d' % idx)

            return label, img
        except Exception as e:
            print(e)
            print(idx)
            print(self.dataset[idx])
            return None

    def get_augmented_data(self, idx):
        try:
            data = self.dataset[idx]
            label = torch.Tensor([self.labels_list.index(data['label'])]).long().squeeze()
            resized_file_name = data['resized_file_name']
            original_file_name = data['original_file_name']
            url = data['url']
            #transform = torchvision.transforms.ToTensor()
            transform = torchvision.transforms.Compose([
                    torchvision.transforms.ColorJitter(),
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation(5),
                    torchvision.transforms.ToTensor()
            ])
            if os.path.isfile(resized_file_name):
                try:
                    img = Image.open(resized_file_name).convert('RGB')
                except OSError:
                    log.warning('Error loading file %s. Rebuilding.' % resized_file_name)
                    os.remove(resized_file_name)
            if not os.path.isfile(resized_file_name):
                # Check if original image is saved
                img = self.get_original_image(original_file_name, url)

                # Get object bounding box
                xmin = data['bounding_box']['xmin']
                xmax = data['bounding_box']['xmax']
                ymin = data['bounding_box']['ymin']
                ymax = data['bounding_box']['ymax']
                # Cropping window
                width = img.size[0] # TODO: Is this right? I just made a guess.
                height = img.size[1]
                min_size = 224
                if (xmax-xmin)*width < min_size:
                    diff = (min_size-(xmax-xmin)*width)/width
                    xmin -= diff/2
                    xmax += diff/2
                if (ymax-ymin)*height < min_size:
                    diff = (min_size-(ymax-ymin)*height)/height
                    ymin -= diff/2
                    ymax += diff/2
                # Ensure we stay in the boundaries
                box_left   = width*xmin
                box_right  = width*xmax
                box_top    = height*ymin
                box_bottom = height*ymax
                if box_left < 0:
                    box_right -= box_left
                    box_left -= box_left
                elif box_right > img.size[0]:
                    box_left += img.size[0]-box_right
                    box_right += img.size[0]-box_right
                if box_top < 0:
                    box_bottom -= box_top
                    box_top -= box_top
                elif box_bottom > img.size[1]:
                    box_top += img.size[1]-box_bottom
                    box_bottom += img.size[1]-box_bottom
                # crop
                img = img.crop([box_left, box_top, box_right, box_bottom])
                img.save(resized_file_name, format='jpeg')
            # Remove channel if alpha
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                print('Found transparency. Processing alpha channels.')
                alpha = img.split()[3]
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=alpha)
                img = background
            if img.mode == 'CMYK':
                img = img.convert('RGB')
            img = transform(img)
            # Add channels if greyscale
            if img.size()[0] == 1:
                img = img.repeat(3,1,1)
            if img.size()[0] == 4:
                print('Alpha channel found, but not processed: %d' % idx)
            # Augmentation
            #augmentation_transform = torchvision.transforms.Compose([
            #        torchvision.transforms.ColorJitter(),
            #        #torchvision.transforms.RandomCrop(),
            #        torchvision.transforms.RandomResizedCrop(224),
            #        torchvision.transforms.RandomHorizontalFlip(),
            #        torchvision.transforms.RandomRotation(5),
            #])
            #img = augmentation_transform(img)
            # Random flip
            # Random rotation
            # Colour jitters
            # Random crop
            # Random resizing
            # Random occlusion (TODO?)
            return label, img
        except Exception as e:
            log.debug(e)
            log.debug(idx)
            log.debug(self.dataset[idx])

    def get_original_image(self, original_file_name, url=None):
        if not os.path.isfile(original_file_name):
            if url is not None:
                urllib.request.urlretrieve(url, original_file_name) 
            else:
                log.debug('File not found and no URL provided.')
                return None
        try:
            return Image.open(original_file_name).convert('RGB')
        except OSError:
            #log.warning('Error loading file %s. Redownloading.' % original_file_name)
            #urllib.request.urlretrieve(url, original_file_name) 
            #img = Image.open(original_file_name)
            log.debug('Error loading file %s. Skipping.' % original_file_name)
            return None

    def compute_mean_std(self):
        original_file_names = set([d['original_file_name'] for d in self.dataset])
        #images = itertools.chain.from_iterable((self.get_original_image(ofn).getdata() for ofn in original_file_names))
        for i,ofn in enumerate(original_file_names):
            img = self.get_original_image(ofn).convert('RGB').getdata()
            mean = np.mean(img, axis=0)
            std = np.std(img, axis=0)
            print(i,mean,std)
            if mean.size != 3:
                break
        for i,img in enumerate(images):
            mean = np.mean(img, axis=0)
            std = np.std(img, axis=0)
            print(i,mean,std)
            if mean.size != 3:
                break

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        # See https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg
        self.layer1 = nn.Sequential(
                #nn.BatchNorm2d(3),
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7//2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(192),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1)
        )
        self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
        )
        self.layer7 = nn.Sequential(
                nn.Linear(in_features=7*7*1024, out_features=4096),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=4096, out_features=7*7*30)
        )

    def forward(self, x):
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)
       x = self.layer5(x)
       x = self.layer6(x)
       x = x.view(-1,7*7*1024)
       x = self.layer7(x)
       x = x.view(-1,30,7,7)
       return x

class YoloClassifier(Yolo):
    def __init__(self, labels=[]):
        super(YoloClassifier, self).__init__()
        self.labels = labels
        self.linear = nn.Linear(in_features=4*4*1024,out_features=len(labels))
        self.softmax = nn.Softmax()

    def forward(self, x):
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)
       x = self.layer5(x)
       x = x.view(-1,4*4*1024)
       x = self.linear(x)
       # x = self.softmax(x)
       return x

def collate(batch):
    d_c = torch.utils.data.dataloader.default_collate
    batch = list(filter(lambda x:x is not None, batch))
    if len(batch) == 0:
        return (torch.Tensor([]), torch.Tensor([]))
    return d_c(batch)

def train():
    device = torch.device('cuda')
    #device = torch.device('cpu')

    # Data
    train = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/', output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset', train=True, label_depth=2)
    test = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/', output_dir='/NOBACKUP/hhuang63/oid/OpenImageDatasetValidation', train=False, label_depth=2)
    train.merge_labels(test)

    dataloader = torch.utils.data.DataLoader(data, batch_size=50, num_workers=5, collate_fn=collate)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=50, num_workers=20, collate_fn=collate, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=50, num_workers=10, collate_fn=collate)

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
        torch.save(net.state_dict(), 'weights/classifier-leaf-%d.pt' % iteration)

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

def train_hierarchical():
    device = torch.device('cuda')

    # Init neural net
    net = YoloClassifier()
    net = net.to(device)

    # Data
    data = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/', output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset')
    data.dataset = data.filter_by_parent(data.dataset, data.description_to_class['Baked goods'])
    train_test_split = 0.9
    train, test = torch.utils.data.random_split(
            data,[int(train_test_split*len(data)), len(data)-int(train_test_split*len(data))])
    dataloader = torch.utils.data.DataLoader(data, batch_size=50, num_workers=5, collate_fn=collate)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=50, num_workers=10, collate_fn=collate, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=50, num_workers=10, collate_fn=collate)

    # Training loop
    opt = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for iteration in range(1000):
        print('Iteration %d\t (Saving...)' % iteration)
        torch.save(net.state_dict(), 'weights/classifier-baked-%d.pt' % iteration)

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
    # TODO: Output/return prediction
    print(description)
    return description


if __name__ == "__main__":
    input_dir = '/NOBACKUP/hhuang63/oid/'
    output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset'
    #train()
    #predict('weights/classifier-leaf-28.pt','875806_R.jpg')

    predict(weights_file_name, img_file_name)
