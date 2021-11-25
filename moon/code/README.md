# boostcamp_pstage10

# Environment
## 1. Docker
```bash
docker run -it --gpus all --ipc=host -v ${path_to_code}:/opt/ml/code -v ${path_to_dataset}:/opt/ml/data placidus36/pstage4_lightweight:v0.4 /bin/bash
```
## 2. Install dependencies
```
pip install -r requirements.txt
```

# Run
## 1. train
`python train.py --model_config ${path_to_model_config} --data_config ${path_to_data_config} --model_name ${model_name_shown_in_wandb}`

## 2. knowledge distillation train
`python train.py --model_config ${path_to_model_config} --data_config ${path_to_data_config} --model_name ${model_name_shown_in_wandb} --parent_cfg ${path_to_parent_cfg} --parent_weights ${path_to_parent_weights}`

    # ex)
    python3 train.py --model configs/model/mobilenetv3.yaml
                     --data configs/data/taco.yaml 
                     --parent_cfg exp/efficient-b0/model.yml
                     --parent_weights exp/efficient-b0/best.pt 
                     --model_name noisy_student_training 

## 2. Noisy Student Training
`python train.py --model_config ${path_to_model_config} --data_config ${path_to_data_config} --model_name ${model_name_shown_in_wandb} --parent_cfg ${path_to_parent_cfg} --parent_weights ${path_to_parent_weights} --noisy_train`

    # ex)
    python3 train.py --model configs/model/mobilenetv3.yaml
                     --data configs/data/taco.yaml 
                     --parent_cfg exp/efficient-b0/model.yml
                     --parent_weights exp/efficient-b0/best.pt 
                     --model_name noisy_student_training 
                     --noisy_train


## 4. inference(submission.csv)
python inference.py --model_config configs/model/mobilenetv3.yaml --weight exp/2021-05-13_16-41-57/best.pt --img_root /opt/ml/data/test --data_config configs/data/taco.yaml3

# Reference
Our basic structure is based on [Kindle](https://github.com/JeiKeiLim/kindle)(by [JeiKeiLim](https://github.com/JeiKeiLim))
