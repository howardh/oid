import csv
import json
from tqdm import tqdm
import os
import dill
import urllib
import urllib.request
import PIL
from PIL import Image

import logging as log

import torch
import torch.nn as nn
import torch.utils.data
import torchvision


def get_food_img_ids_from_human_labels():
    food_img_ids = []
    if os.path.isfile('food_img_ids.pkl'):
        with open('food_img_ids.pkl', 'rb') as f:
            food_img_ids = dill.load(f)
    else:
        with open('/NOBACKUP/hhuang63/oid/train-annotations-human-imagelabels.csv', 'r') as f:
            reader = csv.reader(f)
            for r in tqdm(reader, desc='Extracting Food Image IDs', total=27894290):
                if r[2] in food_keys:
                    food_img_ids.append(r[0])
        with open('food_img_ids.pkl', 'wb') as f:
            dill.dump(food_img_ids, f)
    return food_img_ids

def download_images(urls, output_dir='images'):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for url in tqdm(urls, desc='Downloading Images'):
        file_name = os.path.split(url)[-1]
        file_name = os.path.join(output_dir, file_name)
        if os.path.isfile(file_name):
            continue
        urllib.request.urlretrieve (url, file_name) 

def load_oid_data(food_img_rows, food_img_urls, input_dir='images', output_dir='resized_images'):
    if not os.path.isdir(input_dir):
        raise Exception('Invalid input directory: %s' % input_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    aggregate_file_name = os.path.join(output_dir,'data.pkl')
    #if os.path.isfile(aggregate_file_name):
    #    with open(aggregate_file_name, 'rb') as f:
    #        return dill.load(f)
    transform = torchvision.transforms.ToTensor()
    data = []
    for img_id in tqdm(list(food_img_rows.keys())[:1000]):
        try:
            # input file name
            url = food_img_urls[img_id]
            file_name = os.path.split(url)[-1]
            file_name = os.path.join(input_dir, file_name)
            for i,row in enumerate(food_img_rows[img_id]):
                # label
                label = row[2]
                # output file name
                output_file_name = '%s-%d' % (img_id, i)
                output_file_name = os.path.join(output_dir, output_file_name)
                if os.path.isfile(output_file_name):
                    img = Image.open(output_file_name)
                    img = transform(img)
                    data.append([label,img])
                    continue
                # Load file
                img = Image.open(file_name)
                # Get object centre/width/height
                xmin = float(row[4])
                xmax = float(row[5])
                ymin = float(row[6])
                ymax = float(row[7])
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
                    target_size = target_size/min(img.size)
                # resize
                img = img.resize([int(scale*img.size[0]), int(scale*img.size[1])])
                # Cropping window
                xcentre = (xmin+xmax)/2
                ycentre = (ymin+ymax)/2
                box_left   = img.size[0]*xcentre-224/2
                box_right  = img.size[0]*xcentre+224/2
                box_top    = img.size[1]*ycentre-224/2
                box_bottom = img.size[1]*ycentre+224/2
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
                img.save(output_file_name, format='jpeg')
                img = transform(img)
                data.append([label, img])
        except Exception as e:
            print(e)
    with open(os.path.join(output_dir,'data.pkl'), 'wb') as f:
        dill.dump(data, f)
    return data

class OpenImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir='.', output_dir='OpenImageDataset'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.original_dir = os.path.join(output_dir, 'original')
        self.resized_dir = os.path.join(output_dir, 'resized')

        log.info('Creating OpenImageDataset directories')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.resized_dir, exist_ok=True)

        log.info('Loading class description mappings')
        self.class_to_description, self.description_to_class = self.get_class_descriptions()
        tree = self.get_label_tree()

        food_class = self.description_to_class['Food']
        foods = tree['/m/0bl9f'][food_class]
        self.food_keys = self.extract_label_tree_keys(foods)

        log.info('Loading dataset metadata')
        #food_img_ids = get_food_img_ids_from_human_labels()
        food_img_rows = self.get_food_from_bbox()
        food_img_urls = self.get_food_img_urls(food_img_rows.keys())
        dataset = []
        for img_id in tqdm(sorted(list(food_img_rows.keys()))):
            url = food_img_urls[img_id]
            original_file_name = os.path.split(url)[-1]
            original_file_name = os.path.join(self.original_dir, original_file_name)
            for i,row in enumerate(sorted(food_img_rows[img_id], key=lambda x: x[2])):
                # label
                label = row[2]
                # bounding ox
                xmin = float(row[4])
                xmax = float(row[5])
                ymin = float(row[6])
                ymax = float(row[7])
                # output file name
                resized_file_name = '%s-%d' % (img_id, i)
                resized_file_name = os.path.join(self.resized_dir, resized_file_name)
                # Save data
                dataset.append({
                    'url': url,
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            data = self.dataset[idx]
            label = torch.Tensor([self.food_keys.index(data['label'])]).long().squeeze()
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

    def get_class_description(self, class_id):
        if type(class_id) is torch.Tensor:
            class_id = class_id.item()
        if type(class_id) is int:
            class_id = self.food_keys[class_id]
        return self.class_to_description[class_id]

    def get_food_from_bbox(self):
        food_img_rows = {}
        output_file_name = os.path.join(self.output_dir, 'food_img_bbox.pkl')
        input_file_name = os.path.join(self.input_dir, 'train-annotations-bbox.csv')
        if os.path.isfile(output_file_name):
            with open(output_file_name, 'rb') as f:
                food_img_rows = dill.load(f)
        else:
            with open(input_file_name, 'r') as f:
                reader = csv.reader(f)
                for r in tqdm(reader, desc='Extracting Food Image IDs', total=14610230):
                    if r[2] in self.food_keys:
                        if r[0] in food_img_rows:
                            food_img_rows[r[0]].append(r)
                        else:
                            food_img_rows[r[0]] = [r]
            with open(output_file_name, 'wb') as f:
                dill.dump(food_img_rows, f)
        return food_img_rows

    def get_food_img_urls(self, ids):
        ids = set(ids)
        food_img_urls = {}
        output_file_name = os.path.join(self.output_dir, 'food_img_urls.pkl')
        input_file_name = os.path.join(self.input_dir, 'image_ids_and_rotation.csv')
        if os.path.isfile(output_file_name):
            with open(output_file_name, 'rb') as f:
                food_img_urls = dill.load(f)
        else:
            with open(input_file_name, 'r') as f:
                reader = csv.reader(f)
                for r in tqdm(reader, desc='Extracing photo URLs', total=9178276):
                    # ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation
                    if r[0] in ids:
                        food_img_urls[r[0]]=r[2]
                        ids.remove(r[0])
            with open(output_file_name, 'wb') as f:
                dill.dump(food_img_urls, f)
        return food_img_urls

    def get_class_descriptions(self):
        class_description_filename = os.path.join(self.input_dir, 'class-descriptions.csv')
        with open(class_description_filename) as f:
            reader = csv.reader(f)
            class_to_description = dict(reader)
            description_to_class = dict([v,k] for k,v in class_to_description.items())
        return class_to_description, description_to_class

    def get_label_tree(self):
        """
            Parse json data into a dictionary.
            data = {
                LabelName: "name",
                Subcategory: [data] | None
            }
        """
        file_name = os.path.join(self.input_dir, 'bbox_labels_600_hierarchy.json')
        with open(file_name) as f:
            data = json.load(f)
        def parse(data):
            if 'Subcategory' not in data or data['Subcategory'] is None:
                return {data['LabelName']: None}
            d = dict()
            for x in data['Subcategory']:
                d.update(parse(x))
            return {data['LabelName']: d}
        return parse(data)

    def extract_label_tree_keys(self, node):
        keys = []
        for k,v in node.items():
            keys.append(k)
            if type(v) is dict:
                keys += self.extract_label_tree_keys(v)
        return sorted(keys)

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        # See https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg
        self.layer1 = nn.Sequential(
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
    def __init__(self):
        super(YoloClassifier, self).__init__()
        self.linear = nn.Linear(in_features=4*4*1024,out_features=81)
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

if __name__ == "__main__":
    device = torch.device('cuda')
    #device = torch.device('cpu')

    # Init neural net
    net = YoloClassifier()
    #net.load_state_dict(torch.load('weights/classifier-1.pt'))
    net = net.to(device)

    # Data
    data = OpenImageDataset(input_dir='/NOBACKUP/hhuang63/oid/', output_dir='/NOBACKUP/hhuang63/oid/OpenImageDataset')
    train_test_split = 0.9
    train, test = torch.utils.data.random_split(
            data,[int(train_test_split*len(data)), len(data)-int(train_test_split*len(data))])
    dataloader = torch.utils.data.DataLoader(data, batch_size=50, num_workers=5, collate_fn=collate)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=70, num_workers=10, collate_fn=collate, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=70, num_workers=10, collate_fn=collate)

    # Training loop
    opt = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for iteration in range(1000):
        print('Iteration %d\t (Saving...)' % iteration)
        torch.save(net.state_dict(), 'weights/classifier-%d.pt' % iteration)

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

    def sort_predictions(pred):
        """
        Given a size (1,81) tensor consisting of log softmax outputs, return a new array sorted by the log softmax probability.
        Output consists of the original log softmax outputs and the original index.
        """
        return sorted([(i,p) for i,p in enumerate(pred.detach().cpu().numpy()[0,:])], key=lambda x: x[1], reverse=True)
