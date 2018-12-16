from io import BytesIO
import pdb

import data
from data.labelshierarchy import LabelTree
from data.labelshierarchy import LabelsHierarchy


mock_json_one_node = {
    'LabelName': 'a'
}
mock_json_two_levels = {
    'LabelName': 'a',
    'Subcategory': [
        {'LabelName': 'aa'}
    ]
}
mock_json_three_levels = {
    'LabelName': 'a',
    'Subcategory': [
        {
            'LabelName': 'aa',
            'Subcategory': [
                {'LabelName': 'aaa'}
            ]
        }
    ]
}

def test_label_vector_conversion():
    lh = LabelsHierarchy()
    lh.load_from_json(mock_json_two_levels)
    lh.compute_indices(root='a')
    v = lh.label_to_vector('aa')
    print(v)
    l = lh.vector_to_labels(v[1])
    print(l)
    #pdb.set_trace()
    #assert l['a']['value'] == 1
    assert l['a']['subcategories']['aa']['value'] == 1
