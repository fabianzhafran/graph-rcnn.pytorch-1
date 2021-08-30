## Setup


1. follow `README.md` for installing packages, preparing data, and training the graph-rcnn also using VG datasets (if you have not done the training).
2. On SCC, use this module setting:
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

### Run Detectron
1. make sure to change your custom path in scripts/run_bottom_up.py
2. module load gcc
3. Install detectron2: 
  - Follow installation guide here: https://github.com/airsplay/py-bottom-up-attention.git
4. pip install numpy==1.16.1
5. pip install scikit-image==0.15.0
6. create folder results in directory graph-rcnn
7. create folder results in directory scripts
8. Run the python script
```
cd graph-rcnn.pytorch/scripts
python run_bottom_up.py
```
9. Results are dumped in `graph-rcnn.pytorch/results/`
  - `bottom_up_predictions.pth` will be used in the merging phase

### Merge graph
```
cd graph-rcnn.pytorch/merge_graph
python merge_graph.py
```
