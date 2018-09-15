import os
import csv

class ClassDescriptions(object):
    def __init__(self, input_dir='.', file_name='class-descriptions.csv',
            output_dir='.'):
        self.input_dir = input_dir
        self.file_name = file_name

    def load(self, file_name=None):
        if file_name is None:
            file_name = os.path.join(self.input_dir, self.file_name)
        with open(file_name) as f:
            reader = csv.reader(f)
            self.class_to_description = dict(reader)
            self.class_to_description['/m/0bl9f'] = 'Entity'
            self.description_to_class = dict([v,k] for k,v in self.class_to_description.items())

    def get_class_id(self, desc):
        return self.description_to_class[desc]
    def get_description(self, class_id):
        return self.class_to_description[class_id]
    def get_class_ids(self):
        return set(self.class_to_description.keys())
    def get_descriptions(self):
        return set(self.description_to_class)

    def __getitem__(self, key):
        if key in self.class_to_description:
            return self.class_to_description[key]
        if key in self.description_to_class:
            return self.description_to_class[key]
        raise KeyError('Key not found: %s' % key)

    def __len__(self):
        if self.class_to_description is None:
            raise Exception('Data must be loaded before length can be computed.')
        return len(self.class_to_description)
