import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import pandas as pd

from src.core.imaging.io import read_rgb_image
from src.core.imaging.transforms.functional import resize_image
from src.core.utils.rle import rle_decode

WIDTH = 704
HEIGHT = 520


class CellDataset(Dataset):
    def __init__(self, image_dir: str, df: pd.DataFrame, transforms=None, resize: bool=False):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df

        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(HEIGHT * resize)
            self.width = int(WIDTH * resize)
        else:
            self.height = HEIGHT
            self.width = WIDTH

        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                'image_id': row['id'],
                'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                'annotations': row["annotation"]
            }

    def get_box(self, a_mask):
        """ Get the bounding box of a given mask """
        pos = np.where(a_mask)
        x_min = np.min(pos[1])
        x_max = np.max(pos[1])
        y_min = np.min(pos[0])
        y_max = np.max(pos[0])
        return [x_min, y_min, x_max, y_max]

    def __getitem__(self, idx):
        """ Get the image and the target"""

        img_path = self.image_info[idx]["image_path"]
        img = read_rgb_image(img_path)

        if self.should_resize:
            img = resize_image(img, (self.height, self.width))
            # img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]

        n_objects = len(info['annotations'])
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        boxes = []

        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (HEIGHT, WIDTH))
            # a_mask = Image.fromarray(a_mask)

            if self.should_resize:
                a_mask = resize_image(a_mask, (self.height, self.width))

            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask

            boxes.append(self.get_box(a_mask))

        # dummy labels
        labels = [1 for _ in range(n_objects)]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'is_crowd': is_crowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)
