import os
import csv
import json
import numpy as np
from collections import defaultdict, ChainMap

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

    def __len__(self):
        l = 1
        for subtree in self.subcategories:
            l += len(subtree)
        return l

class LabelsHierarchy(object):
    def __init__(self, input_dir='.',
            file_name='bbox_labels_600_hierarchy.json', output_dir='.'):
        self.input_dir = input_dir
        self.file_name = file_name

        self.tree = None
        self.subtrees = {}
        self.indices = {} # Index range of children of keyed class
        self.parents = {}

        self.vector_length = None
        self.root = None

    def load(self, file_name=None):
        if file_name is None:
            file_name = os.path.join(self.input_dir, 'bbox_labels_600_hierarchy.json')
        self.load_from_file(file_name)

    def load_from_file(self, file_name):
        with open(file_name) as f:
            data = json.load(f)
        self.load_from_json(data)

    def load_from_json(self, data):
        """
            Parse json data into a dictionary.
            data = {
                LabelName: "name",
                Subcategory: [data]
            }
        """
        self.tree = self.parse_json_tree(data)
        self.subtrees = self.extract_all_subtrees(self.tree)
        self.parents = self.extract_parent_relations(self.tree)

    @staticmethod
    def parse_json_tree(data):
        if 'Subcategory' not in data or data['Subcategory'] is None:
            tree = LabelTree(data['LabelName'])
            return tree
        subcategories = []
        for x in data['Subcategory']:
            subcategories.append(LabelsHierarchy.parse_json_tree(x))
        tree = LabelTree(data['LabelName'], subcategories)
        return tree

    @staticmethod
    def extract_parent_relations(tree):
        """ Given a tree, return a dictionary mapping each value in the tree
        to a set of parents.
        """
        output = defaultdict(lambda: set())
        for child in tree.subcategories:
            output[child.key].add(tree.key)
            for k,p in LabelsHierarchy.extract_parent_relations(child).items():
                output[k] |= p
        return output

    @staticmethod
    def extract_all_subtrees(tree):
        """ Given a tree, return a dictionary mapping a key to the key rooted
        on that key.
        """
        output = {tree.key: tree}
        for child in tree.subcategories:
            output = {**output, **LabelsHierarchy.extract_all_subtrees(child)}
        return output

    def __getitem__(self, key):
        """ Return the subtree rooted on `key`. """
        return self.subtrees[key]

    def __len__(self):
        return len(self.subtrees[self.root])

    def compute_indices(self,root=None):
        """
        We have a tree of labels. Labels are grouped by level.
        Return a list where the first two elements are the range of indices
        associated with the label's children, and the third element is a
        dictionary containing the indices of the children relative to the
        starting index.
        """
        def compute(tree, level=0):
            num_children = len(tree.subcategories)

            children_keys = sorted([x.key for x in tree.subcategories])
            output = {tree.key: [
                0, num_children,
                dict([(k,i) for i,k in enumerate(children_keys)])
            ]}

            for i,child in enumerate(tree.subcategories):
                for k,v in compute(child, level+1).items():
                    if k in output:
                        continue
                    output[k] = [num_children+v[0], num_children+v[1], v[2]]
            return output

        if root is None:
            root = self.tree.key
        self.root = root
        indices = {None: [0,1,{root: 0}]}
        for k,v in compute(self.subtrees[root],0).items():
            indices[k] = [1+v[0], 1+v[1], v[2]]
        self.indices = indices

        # Compute max index
        self.vector_length = np.array(
                [(i[0],i[1]) for i in self.indices.values()]
        ).flatten().max()

    def expand_labels(self,label):
        """
        Given a label, return the range of indices containing that label, the
        index of the given label, and the same values for all of its parents.
        For example, given the label "Banana", return a list of objects which
        looks like
            [
                {
                    'label': 'Banana',
                    'range': [10,20],
                    'index': 15
                },{
                    'label': 'Fruit',
                    'range': [1,5],
                    'index': 2
                }
            ]
        """
        output = []
        for p in self.parents[label]:
            if p not in self.indices:
                p = None
            index_range = self.indices[p][:2]
            children_indices = self.indices[p][2]
            output.append({
                    'label': label,
                    'range': index_range,
                    'index': children_indices[label]
            })
            output += self.expand_labels(p)
        return output

    def vector_to_labels(self, vec, label_mapping=None):
        """ Given a vector of values, map each value back to the labels in the
        label tree.
        """
        def convert(tree, parent):
            index_range = self.indices[parent][:2]
            child_indices = self.indices[parent][2]
            index = index_range[0]+child_indices[tree.key]
            output = {
                'value': vec[index],
                'subcategories': dict(ChainMap(*[
                    convert(child, tree.key) for child in tree.subcategories
                ]))
            }
            if label_mapping:
                return {label_mapping[tree.key]: output}
            else:
                return {tree.key: output}

        return convert(self.subtrees[self.root], None)

    def label_to_vector(self, label):
        label = self.expand_labels(label)
        mask = np.zeros(len(self))
        expected_output = np.zeros(len(self))
        for l in label:
            expected_output[l['range'][0]+l['index']] = 1
            mask[l['range'][0]:l['range'][1]] = 1
        return mask, expected_output
