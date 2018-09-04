import os
import csv
import json

class LabelTree():
    def __init__(self, key=None, subcategories=[]):
        self.key = key
        self.subcategories = subcategories

    def __str__(self):
        return '(%s,%s)' % (self.key, self.subcategories)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        yield self.key
        for subtree in self.subcategories:
            for k in subtree:
                yield k

class LabelsHierarchy(object):
    def __init__(self, input_dir='.',
            file_name='bbox_labels_600_hierarchy.json', output_dir='.'):
        self.input_dir = input_dir
        self.file_name = file_name

        self.tree = None
        self.subtrees = None

    def load(self, file_name=None):
        """
            Parse json data into a dictionary.
            data = {
                LabelName: "name",
                Subcategory: [data]
            }
        """
        if file_name is None:
            file_name = os.path.join(self.input_dir, 'bbox_labels_600_hierarchy.json')
        with open(file_name) as f:
            data = json.load(f)
        self.subtrees = {}
        def parse(data):
            if 'Subcategory' not in data or data['Subcategory'] is None:
                tree = LabelTree(data['LabelName'])
                self.subtrees[tree.key] = tree
                return tree
            subcategories = []
            for x in data['Subcategory']:
                subcategories.append(parse(x))
            tree = LabelTree(data['LabelName'], subcategories)
            self.subtrees[tree.key] = tree
            return tree
        self.tree = parse(data)

    def __getitem__(self, key):
        """ Return the subtree rooted on `key`. """
        return self.subtrees[key]
