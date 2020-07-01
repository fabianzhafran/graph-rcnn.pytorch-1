## Setup


1. follow `README.md` for installing packages and preparing data.
2. On SCC, use this module setting:
```
module load cuda/10.1
module load torch/2.1
module load gcc/9.3.0
module load python3/3.7.3
```

## RefCOCO
### Run GraphRCNN
1. Make sure you change to your custom path in `graph-rcnn.pytorch/lib/data/refcoco_dataset.py`
2. Run the eval script: (Make sure you are running in SCC computing node with P100 GPU)
```
cd graph-rcnn.pytorch
sh scripts/eval.sh
```
3. Results are dumped in `graph-rcnn.pytorch/results/`
  - `predictions.pth` and `predictions_pred.pth` will be used in the merging phase.

### Run Detectron
1. Install detectron2: 
  - Follow installation guide here: https://github.com/airsplay/py-bottom-up-attention.git
2. Run the python script
```
cd graph-rcnn.pytorch/scripts
python run_bottom_up.py
```
3. Results are dumped in `graph-rcnn.pytorch/results/`
  - `bottom_up_predictions.pth` will be used in the merging phase

### Merge graph
```
cd graph-rcnn.pytorch/scripts
python merge_graph.py
```
