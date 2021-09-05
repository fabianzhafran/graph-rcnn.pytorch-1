ALGORITHM="sg_imp"
CHECKPOINT="checkpoints/ckpt/sg_imp_step/sg_imp_step_ckpt.pth"
python main.py --config-file configs/sgg_res101_step_refcoco+.yaml --inference --algorithm $ALGORITHM
