import os
import io
import h5py
import json
from tqdm import tqdm
#import deepdish as dd
import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2 import model_zoo

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
save_dir = "../results_extended/"

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

# def get_refer_classes():
#     refer = REFER(dataset='refcoco', data_root='/projectnb/statnlp/gik/refer/data', splitBy='google')
    
#     lastIdx = 1
#     for key, value in refer.Cats.items():
#         lastIdx = max(lastIdx, int(key))
#     list_classes = ['None' for i in range(lastIdx+1)]
#     for key, value in refer.Cats.items():
#         list_classes[int(key)] = value
#     return list_classes
        
def gen_mini_data():
    pass


def showarray(a, fn, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(fn, fmt)
    #display(Image(data=f.getvalue()))

def doit(raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        #print("Original image size: ", (raw_height, raw_width))

        # Preprocessing/
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
#         image = predictor.aug.get_transform(raw_image).apply_image(raw_image)
        new_height, new_width = image.shape[:2]
        #print("Transformed image size: ", image.shape[:2])
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
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        #print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)
        #print("Attr_prob", attr_prob.shape)
        #print("Max_attr_prob", max_attr_prob.shape)

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5): #1.0, 0.1
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label
        instances.attr_logits = attr_prob[ids].detach()
        instances.cls_logits = probs[ids].detach()

        print(instances)

        return instances, roi_features, new_height, new_width


if __name__ == "__main__":
    
    data_path = "/projectnb/statnlp/gik/refer/"

    refcoco_vg_classes = []
    with open(os.path.join(data_path, 'extended_objects_vocab_vg_recoco.txt')) as f:
        for object in f.readlines():
            refcoco_vg_classes.append(object.split(',')[0].lower().strip())

    data_path = "/projectnb/statnlp/gik/py-bottom-up-attention/demo/data/genome/1600-400-20"
    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())
           

    
    MetadataCatalog.get("vg").thing_classes = refcoco_vg_classes
    MetadataCatalog.get("vg").attr_classes = vg_attrs


    NUM_CLASSES = len(refcoco_vg_classes)
    
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
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    predictor = DefaultPredictor(cfg)


    #im = cv2.imread("/projectnb/llamagrp/shawnlin/ref-exp-gen/py-bottom-up-attention/demo/data/images/input.jpg")
    #im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #showarray(im_rgb, save_dir+"raw_img.jpg")

    #hf = h5py.File("./bottom_up_data.h5", "w")
    data = []

    for i, (im, ref_expr, img_id, ann_id, ref_id) in enumerate(tqdm(gen_ref_coco_data())):
        if (i < 10):
            outputs_obj_only = predictor(im)
            print(outputs_obj_only)
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("vg"), scale=1.2)
            out = v.draw_instance_predictions(outputs_obj_only["instances"].to("cpu"))
            showarray(out.get_image()[:, :, ::-1], save_dir+"sampleObj_%i.jpg"%i)
            

        #new_entry = {
        #    "img_id": img_id,
        #    "ref_id": ref_id,
        #    "ann_id": ann_id,
        #    "img_height": None,
        #    "img_width": None,
        #    "boxes": None,
        #    "box_scores": None,
        #    "pred_classes": None,
        #    "attr_logits": None,
        #    "cls_logits": None,
        #}
        #showarray(im, save_dir+"raw_img_%i.jpg" % i)

        instances, features, image_h, image_w = doit(im)
        #new_entry["image_height"] = image_h
        #new_entry["image_width"] = image_w

        pred = instances.to('cpu')
        v = Visualizer(im[:, :, :], MetadataCatalog.get("vg"), scale=1.2)
        v = v.draw_instance_predictions(pred)
        showarray(v.get_image()[:, :, ::-1], save_dir+"pred_%i.jpg"%i)
        #print('instances:\n', instances)
        #print()
        #print('boxes:\n', instances.pred_boxes)
        #print()
        #print('Shape of features:\n', features.shape)

        #pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(features)
        #pred_class_probs = torch.nn.functional.softmax(pred_class_logits, -1)[:, :-1]
        #max_probs, max_classes = pred_class_probs.max(-1)
        #print("%d objects are different, it is because the classes-aware NMS process" % (NUM_OBJECTS - torch.eq(instances.pred_classes, max_classes).sum().item()))
        #print("The total difference of score is %0.4f" % (instances.scores - max_probs).abs().sum().item())

        boxes = instances.pred_boxes.tensor
        boxlist = BoxList(boxes.cpu(), (image_w, image_h), mode="xyxy")
        boxlist.add_field("scores", instances.scores.cpu())
        boxlist.add_field("labels", instances.pred_classes.cpu())
        boxlist.add_field("attr_logits", instances.attr_logits.cpu())
        boxlist.add_field("cls_logits", instances.cls_logits.cpu())
        data.append(boxlist)
        #new_entry["boxes"] = instances.pred_boxes.tensor.cpu().numpy()
        #new_entry["box_scores"] = instances.scores.cpu().numpy()
        #new_entry["pred_classes"] = instances.pred_classes.cpu().numpy()
        #new_entry["attr_logits"] = instances.attr_logits.cpu().numpy()
        #new_entry["cls_logits"] = instances.cls_logits.cpu().numpy()
        #np_dict = np.array(list(new_entry.items()))
        #np.savetxt("./data/%i.npy" % i, np_dict)
        #dd.io.save("./data/%i.h5" % i, new_entry, compression="default")
        #data.append(new_entry)
        #with open("./data/%i.json" % i, "w") as f:

    torch.save(data, "../results_extended/bottom_up_predictions.pth")
    #dd.io.save('vg_bottom_up_data.h5', data, compression="default")
    #hf.create_dataset("vg_bottom_up", data=data)
