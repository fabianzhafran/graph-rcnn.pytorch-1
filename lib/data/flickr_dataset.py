import os
from collections import defaultdict
import numpy as np
import copy
import pickle
import scipy.sparse
from PIL import Image
import h5py, json
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from lib.scene_parser.rcnn.structures.bounding_box import BoxList
from .refcoco_dataset import RefCOCO

import cv2
import sys

class Flickr30K(Dataset):
    def __init__(self, data_dir="/projectnb/statnlp/gik/flickr30k/datasets/flickr30k_images", dataset="flickr30k", split="test", transforms=None):
        assert (split in ["train", "val", "test"])
        images_list = "/projectnb/statnlp/gik/flickr30k-label/multi30k-dataset/data/task1/image_splits/all_images.txt"
        assert os.path.exists(images_list), \
            "cannot find folder {}, please download refcoco data into this folder".format(images_list)
        
        self.info = json.load(open("/projectnb/statnlp/gik/graph-rcnn.pytorch/datasets/vg_bm/VG-SGG-dicts.json", 'r'))
        # add background class
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                               self.class_to_ind[k])
        #cfg.ind_to_class = self.ind_to_classes

        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])
        
        all_data = open(images_list, "r")
        self.images_list = [image_name[:-1] for image_name in all_data.readlines()] # image example : 1000092795.jpg
        
        self.data_dir = data_dir
        self.dataset = dataset
        self.transforms = transforms
        self.split = split

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.images_list[index])
        # print("~~index : {}".format(index))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print("print shape before")
        # print(img.shape)
        # print(torch.is_tensor(img))
        width, height = img.shape[0], img.shape[1]

        # get object bounding boxes, labels and relations
        en_label = self.get_label_for_index("en", index);
        fr_label = self.get_label_for_index("fr", index);
        de_label = self.get_label_for_index("de", index);
        cs_label = self.get_label_for_index("cs", index);
        referent_box = [[34.79, 272.54, 106.72, 80.43]] # dummy target bbox
        #referent_box = [self.refer.Anns[ann_id]["bbox"]]
        target = BoxList(referent_box, (width, height), mode="xyxy")
        target.add_field("ref_sents", [])
        target.add_field("label", None)
        target.add_field("extra_fields", {
                "en_label" : en_label[:-1],
                "fr_label" : fr_label[:-1],
                "de_label" : de_label[:-1],
                "cs_label" : cs_label[:-1]
            })

        if self.transforms is not None: 
            img, target = self.transforms(img, target)
        
        info = { "image_name": int(self.images_list[index][:-4]) }
        # change img to tensor
        # nvm fixed itself idk why
        # img = torch.from_numpy(img)
        # print("print shape after")
        # print(img.shape)
        return img, target, index, int(self.images_list[index][:-4]), info
    
    def get_label_for_index(self, language, index):
        label_path = f"/projectnb/statnlp/gik/flickr30k-label/multi30k-dataset/data/task1/raw/{language}_labels/all_label_{language}.txt"
        opened_file = open(label_path)
        for i, line in enumerate(opened_file):
            if (i == index):
                return line


