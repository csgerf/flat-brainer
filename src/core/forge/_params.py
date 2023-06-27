import warnings


scheduler_dict = {
    "LambdaLR": ["optimizer", "lr_lambda", "last_epoch", "verbose"],
    "MultiplicativeLR": ["optimizer", "lr_lambda", "last_epoch", "verbose"],
    "StepLR": ["optimizer", "step_size", "gamma", "last_epoch", "verbose"],
    "MultiStepLR": ["optimizer", "milestones", "gamma", "last_epoch", "verbose"],
    "ConstantLR": ["optimizer", "factor", "total_iters", "last_epoch", "verbose"],
    "LinearLR": ["optimizer", "start_factor", "end_factor", "total_iters", "last_epoch", "verbose"],
    "ExponentialLR": ["optimizer", "gamma", "last_epoch", "verbose"],
    "SequentialLR": ["optimizer", "schedulers", "milestones", "last_epoch", "verbose"],
    "CosineAnnealingLR": ["optimizer", "T_max", "eta_min", "last_epoch", "verbose"],
    "ChainedScheduler": ["optimizer", "schedulers"],
    "ReduceLROnPlateau": ["optimizer", "mode", "factor", "patience", "threshold", "threshold_mode", "cooldown",
                          "min_lr", "eps", "verbose"],
    "CyclicLR": ["optimizer", "base_lr", "max_lr", "step_size_up", "step_size_down", "mode", "gamma", "scale_fn",
                 "scale_mode", "cycle_momentum", "base_momentum", "max_momentum", "last_epoch", "verbose"],
    "CosineAnnealingWarmRestarts": ["optimizer", "T_0", "T_mult", "eta_min", "last_epoch", "verbose"],
    "OneCycleLR": ["optimizer", "max_lr", "total_steps", "epochs", "steps_per_epoch", "pct_start", "anneal_strategy",
                   "cycle_momentum", "base_momentum", "max_momentum", "div_factor", "final_div_factor", "last_epoch",
                   "verbose"],
    "PolynomialLR": ["optimizer", "total_iters", "power", "last_epoch", "verbose"],
    # Add more scheduler names and arguments as needed...
}

optimizer_dict = {
    'ASGD': ['params', 'lr', 'lambd', 'alpha', 't0', 'weight_decay', 'foreach', 'maximize', 'differentiable'],
    'Adadelta': ['params', 'lr', 'rho', 'eps', 'weight_decay', 'foreach', 'maximize', 'differentiable'],
    'Adagrad': ['params', 'lr', 'lr_decay', 'weight_decay', 'initial_accumulator_value', 'eps', 'foreach', 'maximize',
                'differentiable'],
    'Adam': ['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach', 'maximize', 'capturable',
             'differentiable', 'fused'],
    'AdamW': ['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable',
              'differentiable', 'fused'],
    'Adamax': ['params', 'lr', 'betas', 'eps', 'weight_decay', 'foreach', 'maximize', 'differentiable'],
    'LBFGS': ['params', 'lr', 'max_iter', 'max_eval', 'tolerance_grad', 'tolerance_change', 'history_size',
              'line_search_fn'],
    'NAdam': ['params', 'lr', 'betas', 'eps', 'weight_decay', 'momentum_decay', 'foreach', 'differentiable'],
    # 'Optimizer': ['params', 'defaults'],
    'RAdam': ['params', 'lr', 'betas', 'eps', 'weight_decay', 'foreach', 'differentiable'],
    'RMSprop': ['params', 'lr', 'alpha', 'eps', 'weight_decay', 'momentum', 'centered', 'foreach', 'maximize',
                'differentiable'],
    'Rprop': ['params', 'lr', 'etas', 'step_sizes', 'foreach', 'maximize', 'differentiable'],
    'SGD': ['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'foreach',
            'differentiable'],
    'SparseAdam': ['params', 'lr', 'betas', 'eps', 'maximize']
    }

# import inspect
# import torch.optim as optim
#
# optimizers = {}
#
# for name, obj in inspect.getmembers(optim):
#     if inspect.isclass(obj):
#         args = inspect.signature(obj).parameters.keys()
#         optimizers[name] = list(args)
#
# print(optimizers)

def validate_clean_optimizer_config(name: str, params: dict):
    if name not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer: {name}")

    # Validate the optimizer parameters
    for param in params.keys():
        if param not in optimizer_dict[name]:
            warnings.warn(f"Ignoring invalid optimizer argument '{param}' for optimizer '{name}'", UserWarning)
            del params[param]

    return name, params


def validate_clean_scheduler_config(name: str, params: dict):
    if name not in scheduler_dict:
        raise ValueError(f"Unsupported scheduler: {name}")

    for param in params.keys():
        if param not in scheduler_dict[name]:
            warnings.warn(f"Ignoring invalid argument '{param}' for scheduler '{name}'", UserWarning)
            del params[param]

    return name, params
