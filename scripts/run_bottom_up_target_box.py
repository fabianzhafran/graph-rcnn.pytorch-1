import os
import io
import h5py
import json
from tqdm import tqdm
#import deepdish as dd
import detectron2
import torch

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2 import model_zoo
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances

# import some common libraries
import numpy as np
import cv2
import torch

# Show the image in ipynb
from IPython.display import clear_output, Image, display
import PIL.Image

# Import refcoco wrapper
import sys
sys.path.append("/projectnb/statnlp/gik/refer")
# sys.path.append("/projectnb/llamagrp/shawnlin/ref-exp-gen/dataset/refer2/refer")
#sys.path.append("/projectnb/llamagrp/shawnlin/ref-exp-gen/graph-rcnn.pytorch/")
sys.path.append("../")

from lib.scene_parser.rcnn.structures.bounding_box import BoxList
from refer import REFER

NUM_OBJECTS = 36
save_dir = "../results_target_ref/"

def gen_ref_coco_data():

    dataroot = "/projectnb/statnlp/gik/refer/data"
    dataset = "refcoco"
    refer = REFER(dataroot, dataset, "google")

    ref_ids = refer.getRefIds(split="test")[:]
    print("total ref ids:", len(ref_ids))
    for ref_id in ref_ids[:]:
        ref = refer.Refs[ref_id]
        img_id = ref["image_id"]
        ann_id = ref["ann_id"]
        img_path = os.path.join(refer.IMAGE_DIR, refer.Imgs[img_id]["file_name"])
        img = cv2.imread(img_path)
        #im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ref_expr = "\n".join([s["raw"] for s in ref["sentences"]])
        #print(ref_expr)
        #print("img_id", img_id)

        yield (img, ref_expr, img_id, ann_id, ref_id)
        


def get_refer_classes():
    refer = REFER(dataset='refcoco', data_root='/projectnb/statnlp/gik/refer/data', splitBy='google')
    
    lastIdx = 1
    for key, value in refer.Cats.items():
        lastIdx = max(lastIdx, int(key))
    list_classes = [f'None-{i}' for i in range(lastIdx+1)]
    for key, value in refer.Cats.items():
        list_classes[int(key)] = value
    return list_classes
        
def gen_mini_data():
    pass


def showarray(a, fn, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(fn, fmt)
    #display(Image(data=f.getvalue()))

def doit(raw_image, raw_boxes):
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        #print("Original image size: ", (raw_height, raw_width))

        # Preprocessing/
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        new_height, new_width = image.shape[:2]
        #print("Transformed image size: ", image.shape[:2])
        
        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
    
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        #print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        #print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
#         outputs = FastRCNNOutputs(
#             predictor.model.roi_heads.box2box_transform,
#             pred_class_logits,
#             pred_proposal_deltas,
#             proposals,
#             predictor.model.roi_heads.smooth_l1_beta,
#         )
        pred_class_prob = torch.nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        print(attr_prob.shape)
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes,
            attr_scores = max_attr_prob,
            attr_classes = max_attr_label
        )
        roi_features = feature_pooled
#         print(outputs.predict_probs())
#         probs = outputs.predict_probs()[0]
#         boxes = outputs.predict_boxes()[0]

#         attr_prob = pred_attr_logits[..., :-1].softmax(-1)
#         max_attr_prob, max_attr_label = attr_prob.max(-1)
#         #print("Attr_prob", attr_prob.shape)
#         #print("Max_attr_prob", max_attr_prob.shape)

#         # Note: BUTD uses raw RoI predictions,
#         #       we use the predicted boxes instead.
#         # boxes = proposal_boxes[0].tensor

#         # NMS
#         for nms_thresh in np.arange(0.5): #1.0, 0.1
#             instances, ids = fast_rcnn_inference_single_image(
#                 boxes, probs, image.shape[1:],
#                 score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
#             )
#             if len(ids) == NUM_OBJECTS:
#                 break

#         instances = detector_postprocess(instances, raw_height, raw_width)
#         roi_features = feature_pooled[ids].detach()
#         max_attr_prob = max_attr_prob[ids].detach()
#         max_attr_label = max_attr_label[ids].detach()
#         instances.attr_scores = max_attr_prob
#         instances.attr_classes = max_attr_label
        instances.attr_logits = attr_prob.detach()
        instances.cls_logits = torch.nn.functional.softmax(pred_class_logits).detach()

        print(instances.attr_logits.shape)

        return instances, roi_features, new_height, new_width


if __name__ == "__main__":
    
    data_path = "/projectnb/statnlp/gik/py-bottom-up-attention/demo/data/genome/1600-400-20"
#     data_path = "/projectnb/llamagrp/shawnlin/ref-exp-gen/bottom-up-attention/data/genome/1600-400-20"

#     vg_classes = []
#     with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
#         for object in f.readlines():
#             vg_classes.append(object.split(',')[0].lower().strip())

    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())
           

    refer_classes = get_refer_classes()
    
#     MetadataCatalog.get("vg").thing_classes = vg_classes
    MetadataCatalog.get("vg").thing_classes = refer_classes
    
    MetadataCatalog.get("vg").attr_classes = vg_attrs


    NUM_CLASSES = len(refer_classes)
    
    cfg = get_cfg()
    cfg.merge_from_file("/projectnb/statnlp/gik/py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
    
    cfg.DATALOADER.NUM_WORKERS = 2
    
    cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.BASE_LR = 0.00025 
#     cfg.SOLVER.MAX_ITER = 30000    

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    # VG Weight
#     cfg.MODEL.WEIGHTS = "https://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
    cfg.MODEL.WEIGHTS = "/projectnb/statnlp/gik/refer/output/model_final.pth"
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    predictor = DefaultPredictor(cfg)

    data = []
    
    for i, (im, ref_expr, img_id, ann_id, ref_id) in enumerate(tqdm(gen_ref_coco_data())):
#       extract target box
        with open(os.path.join(f'/projectnb/statnlp/gik/graph-rcnn.pytorch-new/merge_graph/labels/lab_{i}.json')) as json_file:
            label = json.load(json_file)
        refs = [[r] for r in label['ref_sents']]
        bbox = label['bbox'][0]
        x1,y1 = bbox[0], bbox[1]
        x2,y2 = x1 + bbox[2], y1 + bbox[3]
        target_box = np.array([[x1,y1,x2,y2]])
        instances, roi_features, new_h, new_w = doit(im, target_box)
        
        boxes = instances.pred_boxes.tensor
        boxlist = BoxList(boxes.cpu(), (new_w, new_h), mode="xyxy")
        boxlist.add_field("scores", instances.scores.cpu())
        boxlist.add_field("labels", instances.pred_classes.cpu())
        boxlist.add_field("attr_logits", instances.attr_logits.cpu())
        boxlist.add_field("cls_logits", instances.cls_logits.cpu())
        data.append(boxlist)
    torch.save(data, 'results/test_set_target_prediction.pth')
