import os
import csv
from tqdm import tqdm
import dill

class ImageUrls(object):
    def __init__(self, input_dir='.',
            file_name='image_ids_and_rotation.csv', output_dir='.'):
        self.input_dir = input_dir
        self.file_name = file_name
        self.output_dir = output_dir

    def load(self, file_name=None):
        if file_name is None:
            file_name = os.path.join(self.input_dir, self.file_name)
        self.urls = {}
        output_file_name = os.path.join(self.output_dir, 'img_urls.pkl')
        if os.path.isfile(output_file_name):
            with open(output_file_name, 'rb') as f:
                self.urls = dill.load(f)
        else:
            with open(file_name, 'r') as f:
                reader = csv.reader(f)
                for r in tqdm(reader, desc='Extracting photo URLs', total=9178276):
                    # ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation
                    img_id = r[0]
                    original_url = r[2]
                    thumbnail_url = r[10]
                    self.urls[r[0]]=(original_url,thumbnail_url)
            with open(output_file_name, 'wb') as f:
                dill.dump(self.urls, f)

    def __getitem__(self, key):
        """ Return the url associated with the given key. """
        return self.urls[key]
