import os
import pickle
from collections import defaultdict
import cv2
import time
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image, ImageOps
from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

class GazeSeqDataset(Dataset):
    def __init__(self, video_path, n_frames=7):
        self.video_path = video_path
        self.n_frames = n_frames

        # load annotation
        with open(os.path.join(video_path, 'annotation.pickle'), "rb") as f:
            anno_data = pickle.load(f)
            anno_data['index']

        #print(anno_data['index'][0])
        self.bodys = anno_data["bodys"]
        self.heads = anno_data["heads"]
        self.gazes = anno_data["gazes"]
        self.img_index = anno_data['index']

        # abort if no data
        if len(self.gazes) < 1:
            self.valid_index = []
            return

        # extract successive frames
        self.valid_index = []
        for i in range(0, len(self.img_index) - self.n_frames):
            if self.img_index[i] == self.img_index[i] and i < len(self.gazes):
                self.valid_index.append(i)
        self.valid_index = np.array(self.valid_index)



        # image transform for body image
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.valid_index)

    def transform(self, item_allframe):
        image = torch.stack(item_allframe['image'])
        head_dir = np.stack(item_allframe['head_dir']).copy()
        body_dir = np.stack(item_allframe['body_dir']).copy()
        gaze_dir = np.stack(item_allframe['gaze_dir']).copy()


        ret_item = {
            'image': image,
            'head_dir': torch.from_numpy(head_dir),
            'body_dir': torch.from_numpy(body_dir),
            'gaze_dir': torch.from_numpy(gaze_dir),
        }

        return ret_item



    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"index {idx} >= len {len(self)}")

        idx = self.valid_index[idx]
        #print(img_path)
        item_allframe = defaultdict(list)
        for j in range(idx, idx + self.n_frames):
            img_path = os.path.join(self.video_path, f"{self.img_index[j]:06}.jpg")
            img = Image.open(img_path)
            img_ = transform(img)

            item = {
                "image":img_,
                "head_dir": self.heads[idx],
                "body_dir": self.bodys[idx],
                "gaze_dir": self.gazes[idx],
            }
            for k, v in item.items():
                item_allframe[k].append(v)
        
        item_allframe = self.transform(item_allframe)
        return item_allframe

def create_gafa_dataset(exp_names, root_dir='./data/preprocessed', n_frames=7):
    exp_dirs = [os.path.join(root_dir, en) for en in exp_names]

    dset_list = []
    for ed in exp_dirs:
        cameras = sorted(os.listdir(ed))
        for cm in cameras:
            if not os.path.exists(os.path.join(ed, cm, 'annotation.pickle')):
                print(f"annotation.pickle not found in {os.path.join(ed, cm)}")
                continue

            dset = GazeSeqDataset(os.path.join(ed, cm), n_frames=n_frames)

            if len(dset) == 0:
                continue
            dset_list.append(dset)

    print("in create_gafa_dataset")

    return ConcatDataset(dset_list)