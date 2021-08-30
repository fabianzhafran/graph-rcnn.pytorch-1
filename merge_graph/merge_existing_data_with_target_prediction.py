import os
import sys
sys.path.append("../")
sys.path.append("/projectnb/statnlp/gik/refer")
sys.path.append("/projectnb/statnlp/gik/rsa_referring_expression")
from helper import *
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import resize
import pandas as pd

from bbox import BBox2D, XYXY
from bbox.metrics import jaccard_index_2d

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from lib.data.refcoco_dataset import RefCOCO
import pdb
import matplotlib.pyplot as plt


def getRandomColor():
    color = np.random.rand(3)
    return color

# Create iou cost matrix for two sets of predictions

def get_iou_score(b1, b2):

    b1 = BBox2D(b1.numpy(), mode=XYXY)
    b2 = BBox2D(b2.numpy(), mode=XYXY)
    return jaccard_index_2d(b1, b2)


def create_cost_matrix(pred1, pred2, thresh=0.2):
    """
    pred1, pred2: BoxList
    """

    n_bbox1, n_bbox2 = len(pred1), len(pred2)
    #print(n_bbox1, n_bbox2)

    M_cost = np.ones((n_bbox1, n_bbox2)) * 999999.9

    thresh=0.0
    for i in range(n_bbox1):
        for j in range(n_bbox2):
            if pred1.extra_fields["scores"][i] >= thresh and pred2.extra_fields["scores"][j] >= thresh:# and\
                #ind_to_classes[pred1.extra_fields["labels"][i]] == vg_classes[pred2.extra_fields["labels"][j]]:
                iou_score = get_iou_score(pred1.bbox[i], pred2.bbox[j])
            else:
                iou_score = 0.0
            #print(i, j)
            M_cost[i][j] = -1*iou_score
    return M_cost

def generate_csv():

    for idx, hf in tqdm(enumerate(data_iterator())):

        df.to_csv("./scene_graph/%i.tsv" % idx, sep="\t")
        
if __name__ == "__main__":
    bottom_up_preds = torch.load("../results/test_set_target_prediction.pth")
    dataset = RefCOCO(split="test")
    data_loader = DataLoader(dataset, shuffle=False)
    # G-RCNN class/relationship vocabulary
    info = json.load(open("../datasets/vg_bm/VG-SGG-dicts.json", 'r'))
    info['label_to_idx']['__background__'] = 0
    class_to_ind = info['label_to_idx']
    # classes: egg/zebra etc.
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    predicate_to_ind = info['predicate_to_idx']
    predicate_to_ind['__background__'] = 0
    predicate_to_ind = info['predicate_to_idx']
    # predicates: eating, flying, has , holding , wearing etc.
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k:predicate_to_ind[k])
    
    # Bottom-up class/attribute vocabulary
    data_path = "/projectnb/statnlp/gik/py-bottom-up-attention/demo/data/genome/1600-400-20"

    # Attribute csv header
    attr_table_headers = [
        "box_alias",
        "image_id",
        "ann_id",
        "ref_id",
        "salience",
        "x1",
        "y1",
        "w",
        "h"
    ]
    obj_type_list = []
    attr_list = []

    vg_classes = []
    with open(os.path.join("/projectnb/statnlp/gik/refer", 'refcoco_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())
            obj_name = "%s" % object.split(',')[0].lower().strip().replace(" ", "_")
            attr_table_headers.append("TYPE_%s" % obj_name)
            obj_type_list.append(obj_name)
    attr_table_headers.append("TYPE_OOV")
    obj_type_list.append("OOV")
    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())
            attr_name = object.split(',')[0].lower().strip()
            attr_table_headers.append("ATTR_%s" % attr_name)
            attr_list.append(attr_name)
    # keep track of images/refs where the obj in the target box is already detected
    same_obj_detected_ids = []
    for i, (img, target, _, img_info) in enumerate(tqdm(iter(dataset))):
        img_id = img_info["img_id"]
        ann_id = img_info["ann_id"]
        ref_id = img_info["ref_id"]
        
        bot_preds = bottom_up_preds[i]
#       count # of same type detection 
        existing_attr_table_df = pd.read_csv(f'attr_tables/attr_{i}.tsv', encoding='utf-8',sep='\t')
        detected_objs = list(existing_attr_table_df['box_alias'])
        detected_objs = [obj[:obj.index('-')] for obj in detected_objs]
        
        bboxes2 = bot_preds.bbox
        labels2 = bot_preds.extra_fields["labels"]
        scores2 = bot_preds.extra_fields["scores"]
        attr_logits2 = bot_preds.extra_fields["attr_logits"]
        attr_probs = torch.nn.functional.softmax(attr_logits2, dim=1)
        cls_probs = torch.nn.functional.softmax(bot_preds.extra_fields["cls_logits"], dim=1)
        img_h = img.shape[0]
        img_w = img.shape[1]

        #df = pd.DataFrame(columns=attr_table_headers)
        f_attr = open(f'attr_tables_with_target_box/attr_{i}.tsv', "a")

        #for j, (bbox, label, score, cls_prob, attr_prob) in enumerate(zip(bboxes2, labels2, scores2, cls_probs, attr_probs)):
        #    if score < score_thresh: continue
        #    box_alias =
        #    salience = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])/(img_h*img_w)
        #    new_row = [img_id, ann_id, ref_id, salience.item()] + [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])] +\
        #              cls_prob.tolist() + attr_prob.tolist()
        #    df.loc[i] = new_row
        #df.to_csv("test_sg.csv", sep="\t")
        
        score_thresh = 0.0
        # keep track of whether the same object in the target box is already detected.
        same_obj_detected = False
        class_cntr2 = {}
        class_alias_map2 = {} # j: class_alias
        for j, (bbox, label, score, cls_prob, attr_prob) in enumerate(zip(bboxes2, labels2, scores2, cls_probs, attr_probs)):
            if score < score_thresh: continue

            #rand_color = getRandomColor()
            #rect = Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0], bbox[3]-bbox[1],linewidth=2,edgecolor=rand_color, facecolor="none")

            class_name = vg_classes[label.item()]
            class_cntr2[class_name] = class_cntr2.get(class_name, 0) + 1
            same_type_counter = detected_objs.count(class_name) + 1
            # if at least 1 of the same class object(s) is almost overlap with the target detection (eg. 80% overlap), don't add new detection
            if same_type_counter > 1:
                box_data = existing_attr_table_df[['box_alias', 'x1','y1','w','h']]
                with open(os.path.join(f'labels/lab_{i}.json')) as json_file:
                    label = json.load(json_file)
                target_bbox = label['bbox'][0]
                top_5_matches = top_5_match(box_data, target_bbox)
                condition = any([name[:name.index('-')] == class_name and similarity > .6 for _,name,similarity in top_5_matches])
                if condition:
                    same_obj_detected = True
            
            class_alias = "%s-%i" % (class_name, same_type_counter)
            class_alias_map2[j] = class_alias
            #plt.text(bbox[0], bbox[1], "%s: %.1f%%" % (class_alias, score*100), color=rand_color)
            # Add the patch to the Axes
            #ax.add_patch(rect)
            print(class_alias)
            salience = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])/(img_h*img_w)
            new_row = [class_alias, img_id, ann_id, ref_id, salience.item()] + [bbox[0].item(), bbox[1].item(), (bbox[2] - bbox[0]).item(), (bbox[3] - bbox[1]).item()] +\
                      cls_prob.tolist() + attr_prob.tolist()
            #df.loc[j] = new_row
            new_row = [str(l) for l in new_row]
            # only write to file if the same object has not been detected
            if not same_obj_detected:
                f_attr.write("%s\n" % ("\t".join(new_row)))
            else:
                same_obj_detected_ids.append(i)
                
        #plt.show()
        f_attr.close()
    np.save('refs_with_target_obj_already_detected.npy', same_obj_detected_ids)
    print(len(same_obj_detected_ids))
    
    
    
    
