import os
import sys
sys.path.append("../")
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

# Utils

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
    grcnn_preds = torch.load("../results/predictions.pth") # grcnn obj bboxes
    grcnn_rel_preds = torch.load("../results/predictions_pred.pth") # grcnn relationship predictions
    bottom_up_preds = torch.load("../results/bottom_up_predictions.pth") # detectron2 attribute predictions

    # RefCOCO dataset loader
    dataset = RefCOCO(split="test")
    data_loader = DataLoader(dataset, shuffle=False)

    # G-RCNN class/relationship vocabulary
    info = json.load(open("../datasets/vg_bm/VG-SGG-dicts.json", 'r'))
    info['label_to_idx']['__background__'] = 0
    class_to_ind = info['label_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])

    predicate_to_ind = info['predicate_to_idx']
    predicate_to_ind['__background__'] = 0
    predicate_to_ind = info['predicate_to_idx']
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k:predicate_to_ind[k])

    rel_table_headers = [
        "rel_alias",
        "image_id",
        "ann_id",
        "ref_id",
    ]

    for rel_name in ind_to_predicates:
        rel_table_headers.append("REL_%s" % rel_name)

    # Bottom-up class/attribute vocabulary
    data_path = "/projectnb/llamagrp/shawnlin/ref-exp-gen/bottom-up-attention/data/genome/1600-400-20"

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
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())
            obj_name = "%s" % object.split(',')[0].lower().strip().replace(" ", "_")
            attr_table_headers.append("TYPE_%s" % obj_name)
            obj_type_list.append(obj_name)
    # TODO: handle OOV obj type?
    attr_table_headers.append("TYPE_OOV")
    obj_type_list.append("OOV")

    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())
            attr_name = object.split(',')[0].lower().strip()
            attr_table_headers.append("ATTR_%s" % attr_name)
            attr_list.append(attr_name)

    # Merge predictions

    for i, (img, target, _, img_info) in enumerate(tqdm(iter(dataset))):
        preds = grcnn_preds[i]
        rel_preds = grcnn_rel_preds[i]
        #print(preds)
        #print(rel_preds)

        img_id = img_info["img_id"]
        ann_id = img_info["ann_id"]
        ref_id = img_info["ref_id"]

        #plt.imshow(img)
        #ax = plt.gca()
        bboxes1 = preds.bbox
        labels1 = preds.extra_fields["labels"]
        scores1 = preds.extra_fields["scores"]
        class_cntr1 = {}
        class_alias_map1 = {} # j: class_alias

        score_thresh = 0.0

        # Draw object bboxes
        for j, (bbox, label, score) in enumerate(zip(bboxes1, labels1, scores1)):
            if score < score_thresh: continue

            #rand_color = getRandomColor()
            #rect = Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0], bbox[3]-bbox[1],linewidth=2,edgecolor=rand_color, facecolor="none")

            class_name = ind_to_classes[label.item()]
            class_cntr1[class_name] = class_cntr1.get(class_name, 0) + 1
            class_alias = "%s-%i" % (class_name, class_cntr1[class_name])
            class_alias_map1[j] = class_alias
            #plt.text(bbox[0], bbox[1], "%s: %.1f%%" % (class_alias, score*100), color=rand_color)
            # Add the patch to the Axes
            #ax.add_patch(rect)
        #print(class_alias_map1)
        #plt.show()

        bot_preds = bottom_up_preds[i]
        #print(preds)

        #img = resize(img, (preds.size[1], preds.size[0]))
        #plt.figure()
        #plt.imshow(img)
        #ax = plt.gca()
        bboxes2 = bot_preds.bbox
        labels2 = bot_preds.extra_fields["labels"]
        scores2 = bot_preds.extra_fields["scores"]
        attr_logits2 = bot_preds.extra_fields["attr_logits"]
        attr_probs = torch.nn.functional.softmax(attr_logits2, dim=1)
        cls_probs = torch.nn.functional.softmax(bot_preds.extra_fields["cls_logits"], dim=1)
        img_h = img.shape[0]
        img_w = img.shape[1]

        #df = pd.DataFrame(columns=attr_table_headers)
        f_attr = open("attr_tables/attr_%i.tsv" % i, "w+")
        f_attr.write("%s\n" % ("\t".join(attr_table_headers)))

        #for j, (bbox, label, score, cls_prob, attr_prob) in enumerate(zip(bboxes2, labels2, scores2, cls_probs, attr_probs)):
        #    if score < score_thresh: continue
        #    box_alias =
        #    salience = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])/(img_h*img_w)
        #    new_row = [img_id, ann_id, ref_id, salience.item()] + [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])] +\
        #              cls_prob.tolist() + attr_prob.tolist()
        #    df.loc[i] = new_row
        #df.to_csv("test_sg.csv", sep="\t")

        class_cntr2 = {}
        class_alias_map2 = {} # j: class_alias
        for j, (bbox, label, score, cls_prob, attr_prob) in enumerate(zip(bboxes2, labels2, scores2, cls_probs, attr_probs)):
            if score < score_thresh: continue

            #rand_color = getRandomColor()
            #rect = Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0], bbox[3]-bbox[1],linewidth=2,edgecolor=rand_color, facecolor="none")

            class_name = vg_classes[label.item()]
            class_cntr2[class_name] = class_cntr2.get(class_name, 0) + 1
            class_alias = "%s-%i" % (class_name, class_cntr2[class_name])
            class_alias_map2[j] = class_alias
            #plt.text(bbox[0], bbox[1], "%s: %.1f%%" % (class_alias, score*100), color=rand_color)
            # Add the patch to the Axes
            #ax.add_patch(rect)

            salience = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])/(img_h*img_w)
            new_row = [class_alias, img_id, ann_id, ref_id, salience.item()] + [bbox[0].item(), bbox[1].item(), (bbox[2] - bbox[0]).item(), (bbox[3] - bbox[1]).item()] +\
                      cls_prob.tolist() + attr_prob.tolist()
            #df.loc[j] = new_row
            new_row = [str(l) for l in new_row]
            f_attr.write("%s\n" % ("\t".join(new_row)))
        #plt.show()
        f_attr.close()
        #df.to_csv("attr_tables/attr_%i.tsv" % i, sep="\t", index=False)

        # Hungarian allignment for objects in both scenes
        M_cost = create_cost_matrix(preds, bot_preds, thresh=score_thresh)
        row_ind, col_ind = linear_sum_assignment(M_cost)
        #print(row_ind)
        #print(col_ind)
        idx_2_set = {}
        idx_1_set = {}
        to_1_2_map = {}
        for idx_1, idx_2 in zip(row_ind, col_ind):
            if idx_1 in class_alias_map1 and idx_2 in class_alias_map2:
                #print("%s <----> %s" % (class_alias_map1[idx_1], class_alias_map2[idx_2]))
                idx_2_set[idx_2] = True
                idx_1_set[idx_1] = True
                to_1_2_map[idx_1] = idx_2

        #print(idx_1_set)

        # Decode relationships according to matched objects
        # Dump relationship table as tsv
        TOPK = 3
        f_rel = open("rel_tables/rel_%i.tsv" % i, "w+")
        f_rel.write("%s\n" % ("\t".join(rel_table_headers)))
        #df_rel = pd.DataFrame(columns=rel_table_headers)
        for j, (idx_pair, rel_scores) in enumerate(zip(rel_preds.extra_fields["idx_pairs"], rel_preds.extra_fields["scores"])):
            i1, i2 = idx_pair[0].item(), idx_pair[1].item()
            if i1 not in idx_1_set or i2 not in idx_1_set:
                continue
            else:
                top_k_rel = (-rel_scores).argsort()[:TOPK]
                #top_k_rel = top_k_rel[::-1]
                rel_str = ""
                for r in top_k_rel:
                    if r != 0:
                        rel_str += "%s: %.3f%% " % (ind_to_predicates[r], rel_scores[r])
                #print("%s <----> %s\t[%s]" % (class_alias_map2[to_1_2_map[i1]], class_alias_map2[to_1_2_map[i2]], rel_str))

                rel_alias = "(%s,%s)" % (class_alias_map2[to_1_2_map[i1]], class_alias_map2[to_1_2_map[i2]])
                new_row = [rel_alias, img_id, ann_id, ref_id] + rel_scores.tolist()
                new_row = [str(l) for l in new_row]
                #df_rel.loc[j] = new_row
                f_rel.write("%s\n" % ("\t".join(new_row)))
        #df_rel.to_csv("rel_tables/rel_%i.tsv" % i, sep="\t", index=False)
        f_rel.close()


        # TODO: Dump the referent object and bbox into 1 csv
        label_info = target.extra_fields
        label_info["bbox"] = target.bbox.numpy().tolist()
        #print(target.extra_fields)
        #print(label_info)
        json.dump(label_info, open("labels/lab_%i.json" % i, "w+"))

        #input()
