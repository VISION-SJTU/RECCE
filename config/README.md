## Configuration Files

#### Model  Configuration
- We use a yaml file to specify the hyperparameters of a model. All the training logs will be placed in `${project_root}/runs/${model_name}/${experiment_id}`. An example are shown below.

```yaml
model:
  name: Recce  # Model Name
  num_classes: 1 
config:
  lambda_1: 0.1  # balancing weight for L_r
  lambda_2: 0.1  # balancing weight for L_m
  distribute:
    backend: nccl
  optimizer:
    name: adam
    lr: 0.0002
    weight_decay: 0.00001
  scheduler:
    name: StepLR
    step_size: 22500
    gamma: 0.5
  resume: False
  resume_best: False
  id: FF++c40  # Specify a unique experiment id.
  loss: binary_ce  # Loss type, either 'binary_ce' or 'cross_entropy'.
  metric: Acc  # Main metric, either 'Acc', 'AUC', or 'LogLoss'.
  debug: False
  device: "cuda:1"  # NOTE: Used only when testing, annotation this line when training.
  ckpt: best_model_1000  # NOTE: Used only when testing to specify a checkpoint id, annotating this line when training.
data:
  train_batch_size: 32
  val_batch_size: 64
  test_batch_size: 64
  name: FaceForensics
  file: "./config/dataset/faceforensics.yml"  # config file for a dataset
  train_branch: "train_cfg"
  val_branch: "test_cfg"
  test_branch: "test_cfg"
```

- We set different hyper-parameters for the learning rate scheduler according to the used dataset as follows:
  - FaceForensics++: The learning rate is decayed by 0.5 every 10 epochs.
  - Celeb-DF: The learning rate is decayed by 0.5 every 10 epochs.
  - WildDeepfake: The learning rate is decayed by 0.9 every 3000 iterations.
  - DFDC: The learning rate is decayed by 0.5 every 3 epochs.

#### Dataset Configuration
- We also use a yaml file to specify the dataset to load for the experiment. These files are placed under `config/dataset/` subfold.
- Briefly, you should change the `root` parameter according to your storage path. 