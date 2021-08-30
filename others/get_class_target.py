import os
import sys
sys.path.append("../")
sys.path.append("/projectnb/statnlp/gik/refer")
sys.path.append("/projectnb/statnlp/gik/rsa_referring_expression")

import json

import torch
  
from collections import defaultdict


if __name__ == "__main__":
    vg_classes = []
    with open(os.path.join("/projectnb/statnlp/gik/refer", 'refcoco_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())
            
            
    bottom_up_preds = torch.load("../results/test_set_target_prediction.pth")
    
    dict_list_idx_class = defaultdict(list)
    for i in range(5000):
        class_name = vg_classes[bottom_up_preds[i].extra_fields["labels"].item()]
        dict_list_idx_class[class_name].append(i)
        print(i, class_name)
        
    with open('dict_list_idx_class.json', 'w') as fp:
        json.dump(dict_list_idx_class, fp)
