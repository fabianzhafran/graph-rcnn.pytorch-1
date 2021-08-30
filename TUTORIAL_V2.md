## Setup
1. follow `README.md` for installing packages, preparing data
2. follow `README.md` to train the graph-rcnn also using VG datasets (if you have not done the training). Or you can use file `sg_imp_step_ckpt.pth` https://drive.google.com/drive/folders/1KI36pQhz419J9QpPDIrbRakT_AHbTaXJ?usp=sharing, then put that file in "./checkpoints/ckpt/sg_imp_step/sg_imp_step_ckpt.pth"
3. On SCC, use this module setting:
```
module load cuda/10.1
module load anaconda3/5.2.0 # If you use conda, torch version would be : torch/1.4, no need to import python from module
module load python3/3.7.3 # If you use modules
module load torch/2.1 # If you use python3/3.7.3 from modules
module load pytorch/1.1 # For refer
module load gcc/8.3.0 # For building rcnn parser
module load gcc/9.3.0 # For the tutorial below
```

## RefCOCO
### Run GraphRCNN
1. Install pytorch 1.4.0 with cuda 10.1 `conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch` 
2. Make sure you change to your custom path in `graph-rcnn.pytorch/lib/data/refcoco_dataset.py`
3. Run the eval script: (Make sure you are running in SCC computing node with P100 GPU)
```
cd graph-rcnn.pytorch
sh scripts/eval.sh
```
4. Results are dumped in `graph-rcnn.pytorch/results/`
  - `predictions.pth` and `predictions_pred.pth` will be used in the merging phase.

### Detectron
#### Setup
1. make sure to change your custom path in scripts/run_bottom_up.py
2. module load gcc
3. Install detectron2: 
  - Follow installation guide here: https://github.com/airsplay/py-bottom-up-attention.git
4. `conda install numpy==1.16.1 scikit-image==0.15.0`    
6. create folder results in directory graph-rcnn
7. create folder results in directory scripts

#### Run detectron
A. Option 1: Using finetuned model
1. In file `run_bottom_up.py` make sure you load the correct dataset (in function `get_refer_dicts` and `get_refer_classes`, in code `refer = REFER(dataset='refcoco', data_root='./data', splitBy='google')`)
2. make sure the data path is correct (`data_path = <path>/py-bottom-up-attention/demo/data/genome/1600-400-20`). This is used for loading the attributes list
3. make sure the config that you use for finetuning is the same with the one that you use for prediction (cfg.merge_from_file(...) and also other cfg setter)
4. use the correct vocabulary (look at the code `MetadataCatalog.get("vg").thing_classes = refer_classes`)
5. use the correct model weights (look at the code `cfg.MODEL.WEIGHTS = ...`)
6. run `python run_bottom_up.py`
7. copy the result `results/bottom_up_predictions.pth` to `../results`
8. do the same thing from step 1-5 in file `run_bottom_up_target_box.py`
9. run `python run_bottom_up_target_box.py`
10. copy the result `results/test_set_target_prediction.pth` to `../results`

B. Option 2: Using pretrained model 
1. In file `run_bottom_up.py` use the vocabulary from pretrained model (`MetadataCatalog.get("vg").thing_classes = vg_classes`)
2. make sure the data path is correct (`data_path = <path>/py-bottom-up-attention/demo/data/genome/1600-400-20`). This is used for loading the attributes list and the type list (vocabulary)
3. use the correct model weights (code `cfg.MODEL.WEIGHTS = "https://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"`)
4. run `python run_bottom_up.py`
5. copy the result `results/bottom_up_predictions.pth` to `../results`
6. do the same thing from step 1-5 in file `run_bottom_up_target_box.py`
7. run `python run_bottom_up_target_box.py`
8. copy the result `results/test_set_target_prediction.pth` to `../results`

### Merge graph (combine graph-rcnn result with detectron2 result)
1. go to folder merge_graph
2. create folder `attr_tables`, `labels`, `rel_tables`, `attr_tables_with_target_box`
3. run `python merge_graph.py`
4. copy the resulted tsv files in folder `attr_tables` to folder `attr_tables_with_target_box`
5. run `python merge_existing_data_with_target_prediction.py`
6. results are in folder `attr_tables`, `labels`, `rel_tables`, `attr_tables_with_target_box`
