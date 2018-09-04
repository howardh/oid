import os
import csv
from tqdm import tqdm

class BoundingBoxes():
    def __init__(self, input_dir='.', file_name='train-annotations-bbox.csv',
            output_dir='.'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_name = file_name

    def load(self, input_file_name=None, output_file_name=None):
        data = {}
        input_file_name = os.path.join(self.input_dir, self.file_name)
        output_file_name = os.path.join(self.output_dir, 'bbox_by_class.pkl')
        if os.path.isfile(output_file_name):
            with open(output_file_name, 'rb') as f:
                data = dill.load(f)
        else:
            with open(input_file_name, 'r') as f:
                reader = csv.reader(f)
                for r in tqdm(reader, desc='Sorting image bounding boxes by class', total=14610230):
                    class_id = r[2]
                    img_id = r[0]
                    if class_id in data:
                        data[class_id].append(r)
                    else:
                        data[class_id] = [r]
            with open(output_file_name, 'wb') as f:
                dill.dump(data, f)
        self.data = data
