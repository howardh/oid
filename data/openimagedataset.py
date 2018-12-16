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
    def __init__(self, input_dir='.', output_dir='OpenImageDataset',
            label_depth=-1, label_root=None, train=True):
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
        labels_hierarchy.compute_indices(root=class_descriptions[label_root])

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
        food_boxes = bounding_boxes[food_labels]

        # TODO: Map from image id to a list of bounding boxes for that image
        # Name images to include bounding box coordinates. This is unambiguous, and no ordering is required.

        dataset = []
        for row in tqdm(bounding_boxes[food_labels], desc='Creating Dataset'):
            img_id = row[0]
            # label
            label = row[2]
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
        self.labels_hierarchy = labels_hierarchy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.train:
            return self.get_augmented_data(idx)
        else:
            return self.get_testing_data(idx)

    def merge_labels(self, dataset):
        """ Given another OpenImageDataset, merge the labels from both so that both datasets have the same labels and label ordering. """
        self.labels_hierarchy = dataset.labels_hierarchy

    def get_num_labels(self):
        return len(self.labels_list)

    def get_testing_data(self, idx):
        try:
            data = self.dataset[idx]
            mask, expected_output = self.labels_hierarchy.label_to_vector(data['label'])
            mask = torch.Tensor(mask)
            expected_output = torch.Tensor(expected_output)

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

            return {'mask': mask, 'output': expected_output}, img
        except Exception as e:
            log.debug(e)
            log.debug(idx)
            log.debug(self.dataset[idx])
            return None

    def get_augmented_data(self, idx):
        try:
            data = self.dataset[idx]
            #label = torch.Tensor([self.labels_list.index(data['label'])]).long().squeeze()
            mask, expected_output = self.labels_hierarchy.label_to_vector(data['label'])
            mask = torch.Tensor(mask)
            expected_output = torch.Tensor(expected_output)

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
            return {'mask': mask, 'output': expected_output}, img
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

def collate(batch):
    d_c = torch.utils.data.dataloader.default_collate
    batch = list(filter(lambda x:x is not None, batch))
    if len(batch) == 0:
        return ({'mask': torch.Tensor([]), 'output': torch.Tensor([])}, torch.Tensor([]))
    return d_c(batch)

