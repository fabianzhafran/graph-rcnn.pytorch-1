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

import cv2
import sys
sys.path.append("/projectnb/llamagrp/gik/refer")

from refer import REFER

class RefCOCO+(Dataset):
    def __init__(self, data_dir="/projectnb/llamagrp/gik/refer/data", dataset="refcoco+", split="train", splitBy="unc", transforms=None):
        assert (split in ["train", "val", "test"])
        assert os.path.exists(data_dir), \
            "cannot find folder {}, please download refcoco+ data into this folder".format(data_dir)

        self.data_dir = data_dir
        self.dataset = dataset
        self.transforms = transforms
        self.split = split
        self.splitBy = splitBy

        self.refer = REFER(self.data_dir, self.dataset, self.splitBy)
        self.ref_ids = self.refer.getRefIds(split=self.split)[:]

        #self.filter_non_overlap = filter_non_overlap
        #self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'

        # read in dataset from a h5 file and a dict (json) file
        #self.im_h5 = h5py.File(self.image_file, 'r')
        self.info = json.load(open("/projectnb/llamagrp/gik/graph-rcnn.pytorch/datasets/vg_bm/VG-SGG-dicts.json", 'r'))
        #self.im_refs = self.im_h5['images'] # image data reference
        #im_scale = self.im_refs.shape[2]

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
        #cfg.ind_to_predicate = self.ind_to_predicates

        #self.split_mask, self.image_index, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
        #    self.roidb_file, self.image_file,
        #    self.split, num_im, num_val_im=num_val_im,
        #    filter_empty_rels=filter_empty_rels,
        #    filter_non_overlap=filter_non_overlap and split == "train",
        #)

        #self.json_category_id_to_contiguous_id = self.class_to_ind

        #self.contiguous_category_id_to_json_id = {
        #    v: k for k, v in self.json_category_id_to_contiguous_id.items()
        #}

    #@property
    #def coco(self):
    #    """
    #    :return: a Coco-like object that we can use to evaluate detection!
    #    """
    #    anns = []
    #    for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
    #        for cls, box in zip(cls_array.tolist(), box_array.tolist()):
    #            anns.append({
    #                'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
    #                'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
    #                'category_id': cls,
    #                'id': len(anns),
    #                'image_id': i,
    #                'iscrowd': 0,
    #            })
    #    fauxcoco = COCO()
    #    fauxcoco.dataset = {
    #        'info': {'description': 'ayy lmao'},
    #        'images': [{'id': i} for i in range(self.__len__())],
    #        'categories': [{'supercategory': 'person',
    #                           'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
    #        'annotations': anns,
    #    }
    #    fauxcoco.createIndex()
    #    return fauxcoco

    #def _im_getter(self, idx):
    #    w, h = self.im_sizes[idx, :]
    #    ridx = self.image_index[idx]
    #    im = self.im_refs[ridx]
    #    im = im[:, :h, :w] # crop out
    #    im = im.transpose((1,2,0)) # c h w -> h w c
    #    return im

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        """
        get dataset item
        """
        # get image
        print("~~~printing from refcoco+_dataset.py~~~")
        ref = self.refer.Refs[self.ref_ids[index]]
        img_id = ref["image_id"]
        ann_id = ref["ann_id"]
        img_path = os.path.join(self.refer.IMAGE_DIR, self.refer.Imgs[img_id]["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("{}. =========".format(index))
        print("img shape type before:")
        print(img.shape)
        #print("img original size", img.shape)
        width, height = img.shape[0], img.shape[1]
        #print("img after size", img.shape)
        ref_expr = "\n".join([s["raw"] for s in ref["sentences"]])

        # get object bounding boxes, labels and relations
        #obj_boxes = [[34.79, 272.54, 106.72, 80.43]] # dummy target bbox
        referent_box = [self.refer.Anns[ann_id]["bbox"]]
        target_raw = BoxList(referent_box, (width, height), mode="xyxy")
        # print(target_raw)
        if self.transforms is not None:
            img, target = self.transforms(img, target_raw)
        else:
            img, target = img, target_raw
        target.add_field("ref_sents", [s["raw"] for s in ref["sentences"]])
        target.add_field("label", self.refer.Anns[ann_id]["category_id"])
        #target.add_field("labels", torch.from_numpy(obj_labels))
        #target.add_field("pred_labels", torch.from_numpy(obj_relations))
        #target.add_field("relation_labels", torch.from_numpy(obj_relation_triplets))
        target = target.clip_to_image(remove_empty=False)

        print("img shape type after:")
        print(img.shape)
        print(type(img))
        print("~~~~~~")
        print()
        info = {"img_id":img_id, "ann_id":ann_id, "ref_id": self.ref_ids[index], "ref_sents": [s["raw"] for s in ref["sentences"]]}

        return img, target, index, info

    #def get_groundtruth(self, index):
    #    width, height = self.im_sizes[index, :]
    #    # get object bounding boxes, labels and relations

    #    obj_boxes = self.gt_boxes[index].copy()
    #    obj_labels = self.gt_classes[index].copy()
    #    obj_relation_triplets = self.relationships[index].copy()

    #    if self.filter_duplicate_rels:
    #        # Filter out dupes!
    #        assert self.split == 'train'
    #        old_size = obj_relation_triplets.shape[0]
    #        all_rel_sets = defaultdict(list)
    #        for (o0, o1, r) in obj_relation_triplets:
    #            all_rel_sets[(o0, o1)].append(r)
    #        obj_relation_triplets = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
    #        obj_relation_triplets = np.array(obj_relation_triplets)

    #    obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))

    #    for i in range(obj_relation_triplets.shape[0]):
    #        subj_id = obj_relation_triplets[i][0]
    #        obj_id = obj_relation_triplets[i][1]
    #        pred = obj_relation_triplets[i][2]
    #        obj_relations[subj_id, obj_id] = pred

    #    target = BoxList(obj_boxes, (width, height), mode="xyxy")
    #    target.add_field("labels", torch.from_numpy(obj_labels))
    #    target.add_field("pred_labels", torch.from_numpy(obj_relations))
    #    target.add_field("relation_labels", torch.from_numpy(obj_relation_triplets))
    #    target.add_field("difficult", torch.from_numpy(obj_labels).clone().fill_(0))
    #    return target

    def get_img_info(self, img_id):

        ref = self.refer.Refs[self.ref_ids[index]]
        img_id = ref["image_id"]
        w, h = self.refer.Imgs[img_id]["width"], self.refer.Imgs[img_id]["height"]
        return {"height": h, "width": w}

    #def map_class_id_to_class_name(self, class_id):
    #    return self.ind_to_classes[class_id]

if __name__ == "__main__":
    ds = RefCOCO+()
    print(ds.next())
