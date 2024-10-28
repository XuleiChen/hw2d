wandb:
  project: hw2d
  group: ffno/grid_sizes/64
  tags:
    - pde
    - hw2d
    - fourier
  notes: ""
  log_model: all
builder:
  _target_: fourierflow.builders.hw2d.HasegawaWakataniBuilder
  train_data_path: "/scratch/xc2695/hw2d_numerical_solver/hw2d_solver/results/hasegawa_wakatani_simulation_data_train.h5"
  valid_data_path: "/scratch/xc2695/hw2d_numerical_solver/hw2d_solver/results/hasegawa_wakatani_simulation_data_validation.h5"
  test_data_path: "/scratch/xc2695/hw2d_numerical_solver/hw2d_solver/results/hasegawa_wakatani_simulation_data_test.h5"
  batch_size: 32
  num_workers: 48
  pin_memory: True # true when using GPU
  feature: 'phi'
#  time_steps: 100 # total time of numerical solver
  k: 1 # Specify the down sampling step size here
  traj_len: 20
routine:
  _target_: fourierflow.routines.Grid2DMarkovExperiment
  conv:
    _target_: fourierflow.modules.FNOFactorized2DBlock
    modes: 16
    width: 64
    n_layers: 4
    input_dim: 3 # not using fourier encoding in use_position,so +2; append_mu, +1
    share_weight: true
    ff_weight_norm: true
  append_mu: False
  use_position: True
  pred_path: "/scratch/xc2695/hw2d_fourierflow/hw2d_fourierflow_predictions.h5"
  optimizer:
    _target_: functools.partial
    _args_: [ "${get_method: torch.optim.AdamW}" ]
    lr: 0.0025
    weight_decay: 0.0001
  scheduler:
    scheduler:
      _target_: functools.partial
      _args_: [ "${get_method: fourierflow.schedulers.CosineWithWarmupScheduler}" ]
      num_warmup_steps: 200
      num_training_steps: 13333
      num_cycles: 0.5
    name: learning_rate
trainer:
  accelerator: gpu
  devices: 1
  precision: 32
  max_epochs: 10
  log_every_n_steps: 100
callbacks:
  - _target_: fourierflow.callbacks.CustomModelCheckpoint
    filename: "{epoch}-{step}-{valid_time_until:.3f}"
    save_top_k: 1
    save_last: true
    monitor: valid_loss
    mode: min
    every_n_train_steps: null
    every_n_epochs: 1
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: step
  - _target_: pytorch_lightning.callbacks.ModelSummary
    max_depth: 4