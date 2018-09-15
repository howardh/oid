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

    def get_map_to_level(self, level=1, ignore_partial_paths=False):
        """ Return a dictionary mapping children to their ancestor at level
        `level`. Level 0 refers to the root node. """
        if level == 0:
            return dict([(k, self.key) for k in self])
        else:
            # Loop through children and assign them all ot their direct parent
            if len(self.subcategories) == 0:
                if ignore_partial_paths:
                    return {}
                else:
                    return {self.key: self.key}
            else:
                output = dict()
                for subtree in self.subcategories:
                    output.update(subtree.get_map_to_level(level-1))
                return output

class LabelsHierarchy(object):
    def __init__(self, input_dir='.',
            file_name='bbox_labels_600_hierarchy.json', output_dir='.'):
        self.input_dir = input_dir
        self.file_name = file_name

        self.tree = None

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
