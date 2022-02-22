# Classifying PH from ECHOS

Data and training pipeline for classification of newborn echocardiograms,
for detection of pulmonary hypertension (PH), a functional heart defect,
in newborns.
<!--
- Main code and classes are located in the *ehco_ph* package/module.

- Scripts for pre-processing data, generating index files (for splitting into train 
and valid), training the networks, and analysing results is found inside 
the *scripts* directory.

To use the echo_ph package, you must run: 
    <code> pip install -e . </code>

Description of main scripts and how to run them:
1) Training (script: scripts/train_simple.py): 
  - training base parameters: 
    - <code> python scripts/train_simple.py --max_epochs 300 --wd 1e-3 --class_balance_per_epoch --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --augment --aug_type 4 --optimizer adamw --pretrained --num_rand_frames 10  </code>
  - other parameters that vary:  
    - <code> --model model_type --batch_size batch_size --label_type label_type --temporal --view view1 view2 ... </code>
    - The temporal model uses batch size of 8, but spatial model a batch size of 64
2) Get metrics from results file:
   - <code> python scripts/evaluation/get_metrics.py --res_dir res_dir</code>
   - add   <code> --multi_class </code> if any of the models from res_dir are not binary classification
   - Note that the res_dir should be the directory storing the directory of other model(s) results dirs.
3) Visualisations:
   - Visualise temporal model: 
        - Save 1 clip per video: 
        - <code> python scripts/visualisations/vis_grad_cam_temp.py --model_path  path_to_trained_model.pt  --model model_type --num_rand_samples 1 --save_video_clip </code>
        - Save full video (feed all frames - but model not trained with that input): <code> python scripts/visualise/vis_grad_cam_temp.py --model_path  path_to_trained_model.pt  --model model_type --all_frames --save_video </code>
   - Visualise spatial model: (TODO: prob. need to adjust also (!)):
     -  <code> python scripts/visualisations/vis_grad_cam.py --todo </code> (TODO: FINISH)
   - Visualisation saliency model: Requires a separate model.
4) Multi-View Majority Vote (MV) / FRAME-LEVEL Joining :
    - <code> MV: python scripts/evaluation/multi_view_ensemble.py base_res_dir --res_files file_name_KAPAP file_name_CV file_name_CV --views kapap cv la </code>
    - <code> Frame-level: python scripts/evaluation/join_view_models_frame_level.py base_res_dir --res_files file_name_KAPAP file_name_CV file_name_CV --views kapap cv la </code>
    - Note: Also some draft files on same topic named 'analyse_result_files.py', and 'analyse_result_files2.py'
     - TODO: Check if I need anything from these files and incorporate it into the multi_view_ensemble.py 
-->
