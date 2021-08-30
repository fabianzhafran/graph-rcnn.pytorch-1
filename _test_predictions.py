import torch
import json
import os


grcnn_preds = torch.load("../graph-rcnn.pytorch/results_flickr30k/predictions.pth") # grcnn obj bboxes
grcnn_rel_preds = torch.load("../graph-rcnn.pytorch/results_flickr30k/predictions_pred.pth") # grcnn relationship predictions
bottom_up_preds = torch.load("../graph-rcnn.pytorch/results_flickr30k/bottom_up_predictions.pth") # detectron2 attribute predictions

# G-RCNN class/relationship vocab
info = json.load(open("../graph-rcnn.pytorch/datasets/vg_bm/VG-SGG-dicts.json", 'r'))
info['label_to_idx']['__background__'] = 0
class_to_ind = info['label_to_idx']
ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])

# Bottom-up class/attribute vocab
data_path = "/projectnb/statnlp/gik/py-bottom-up-attention/demo/data/genome/1600-400-20"
vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

if __name__ == "__main__":
    for i in range(10):
        print("~~~~ ~~~~")
        print("grcnn_preds[{}] labels :".format(i))
        grcnn_classes = []
        for id in grcnn_preds[i].extra_fields["labels"]:
            grcnn_classes.append(ind_to_classes[id])
        print(grcnn_classes)
        # print(grcnn_preds[0].extra_fields["labels"])
        # print("~~~grcnn_rel_preds[0] extra fields~~~")
        # print(grcnn_rel_preds[0].extra_fields)
        bottom_up_classes = []
        print("bottom_up_preds[{}] labels :".format(i))
        for id in bottom_up_preds[i].extra_fields["labels"]:
            bottom_up_classes.append(vg_classes[id])
        print(bottom_up_classes)
    # print(bottom_up_preds[0].extra_fields["labels"])
