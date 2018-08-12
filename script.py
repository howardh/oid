import csv
import json
from tqdm import tqdm
import os

class_description_filename = '/NOBACKUP/hhuang63/oid/class-descriptions.csv'
with open(class_description_filename) as f:
    reader = csv.reader(f)
    class_to_description = dict(reader)
    description_to_class = dict([v,k] for k,v in class_to_description.items())
food_class = description_to_class['Food']

with open('/NOBACKUP/hhuang63/oid/bbox_labels_600_hierarchy.json') as f:
    j = json.load(f)

def parse(data):
    """
        Parse json data into a dictionary.
        data = {
            LabelName: "name",
            Subcategory: [data] | None
        }
    """
    if 'Subcategory' not in data or data['Subcategory'] is None:
        return {data['LabelName']: None}
    d = dict()
    for x in data['Subcategory']:
        d.update(parse(x))
    return {data['LabelName']: d}

def extract_keys_recursively(d):
    keys = []
    for k,v in d.items():
        keys.append(k)
        if type(v) is dict:
            keys += extract_keys_recursively(v)
    return keys

d = parse(j)
foods = d['/m/0bl9f'][food_class]
k = extract_keys_recursively(foods)

food_img_ids = []
if os.path.isfile('food_img_ids.pkl'):
    with open('food_img_ids.pkl', 'rb') as f:
        food_img_ids = dill.load(f)
else:
    with open('/NOBACKUP/hhuang63/oid/train-annotations-human-imagelabels.csv', 'r') as f:
        reader = csv.reader(f)
        for r in tqdm(reader, desc='Extracting Food Image IDs', total=27894290):
            if r[2] in k:
                food_img_ids.append(r[0])
    with open('food_img_ids.pkl', 'wb') as f:
        dill.dump(food_img_ids, f)

food_img_urls = []
with open('/NOBACKUP/hhuang63/oid/image_ids_and_rotation.csv', 'r') as f:
    reader = csv.reader(f)
    for r in tqdm(reader, desc='Extracing photo URLs', total=9178276):
        # ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation
        if r[0] in food_img_ids:
            food_img_urls.append(r[2])
with open('food_img_urls.pkl', 'wb') as f:
    dill.dump(food_img_urls, f)
