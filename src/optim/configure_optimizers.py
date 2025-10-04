import logging

import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from ..optim.lamb import Lamb

logger = logging.getLogger(__name__)


def configure_bert_optimizers(
    model: torch.nn.Module,
    opt_name: str,
    max_lr: float,
    betas: tuple,
    eps: float,
    weight_decay: float,
    steps: int,
    warmup_steps_ratio: float,
    scheduler_names: list,
    blacklist_weight_modules: list = [],
) -> dict:
    """
    Configure the optimizer and learning rate scheduler for BERT models.

    Parameters
    ----------
    parameters : torch.nn.Module
        The model to configure the optimizer for.
    opt_name : str
        The name of the optimizer to use. This should be one of "adamw" or "lamb".
    max_lr : float
        The maximum learning rate to use.
    betas : tuple
        The beta values for the optimizer.
    eps : float
        The epsilon value for the optimizer.
    weight_decay : float
        The weight decay value for the optimizer.
    steps : int
        The total number of training steps.
    warmup_steps_ratio : float
        The ratio of steps to use for the warmup.
    scheduler_names : list
        A list of strings specifying the type of learning rate scheduler to use.
        The first string is for the warmup, the second is for the main training.
    blacklist_weight_modules : list, optional
        A list of strings specifying the names of modules to not apply weight decay to.
        The default is []. This is useful for applying weight decay to only the
        convolutional layers.

    Returns
    -------
    opt_dict : dict
        A dictionary containing the optimizer and learning rate scheduler.

    """
    # Validate the optimizer name
    if opt_name not in ["adamw", "lamb"]:
        raise TypeError(f"Invalid optimizer name: {opt_name}")
    # Select the optimizer
    if opt_name == "adamw":
        optim_func = torch.optim.AdamW
    elif opt_name == "lamb":
        optim_func = Lamb

    # Configure the optimizer
    if blacklist_weight_modules:
        optimizer = _configure_weight_decay_groups(
            optim_func,
            model,
            max_lr,
            betas,
            eps,
            weight_decay,
            blacklist_weight_modules,
        )
    else:
        optimizer = optim_func(
            model.parameters(),
            lr=max_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    # Configure the first part of the learning rate schedule
    scheduler_1 = _configure_scheduler_1(
        optimizer, warmup_steps_ratio, steps, scheduler_names[0]
    )

    # Configure the second part of the learning rate schedule
    scheduler_2 = _configure_scheduler_2(
        optimizer, steps, warmup_steps_ratio, scheduler_names[1]
    )

    # Configure scheduler
    scheduler = SequentialLR(
        optimizer,
        [scheduler_1, scheduler_2],
        milestones=[int(warmup_steps_ratio * steps)],
    )

    # Configure optimizer dictionary
    opt_dict = {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }

    return opt_dict


def _configure_weight_decay_groups(
    optim_func: torch.optim.Optimizer,
    model: torch.nn.Module,
    max_lr: float,
    betas: tuple,
    eps: float,
    weight_decay: float,
    blacklist_weight_modules: list = ["norm", "embedding"],
):
    """
    Modified approach for configuring AdamW that excludes various parameters from the weight decay.
    Source: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.

    Parameters
    ----------
    optim_func : torch.optim.Optimizer
        The optimizer function to use.
    parameters : torch.nn.Module
        The model to configure the optimizer for.
    max_lr : float
        The maximum learning rate to use.
    betas : tuple
        The beta values for the AdamW optimizer.
    eps : float
        The epsilon value for the AdamW optimizer.
    weight_decay : float
        The weight decay value for the AdamW optimizer.
    blacklist_weight_modules : list, optional
        A list of strings specifying the names of modules to not apply weight decay to.
        The default is ["norm", "embedding"]. This is useful for applying weight decay to only the
        convolutional layers.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The optimizer object.

    """
    # Separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

            if pn.endswith("bias"):
                # All biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and any(
                x in fpn for x in blacklist_weight_modules
            ):
                # Weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            else:
                # Weights of remaining modules will be weight decayed
                decay.add(fpn)

    # Validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # Log the weights that will not experience weight decay
    logger.info(
        f"Weight decay will not be applied to the following modules: {no_decay}"
    )

    # Create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim_func(optim_groups, lr=max_lr, betas=betas, eps=eps)

    return optimizer


def _configure_scheduler_1(optimizer, warmup_steps_ratio, steps, scheduler_name):
    # Configure the first part of the learning rate schedule
    if scheduler_name == "linear":
        scheduler_1 = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1,
            total_iters=int(warmup_steps_ratio * steps),
        )
    elif scheduler_name == "constant":
        # configure second lr scheduler
        scheduler_1 = ConstantLR(
            optimizer,
            factor=1,
            total_iters=int(warmup_steps_ratio * steps),
        )
    else:
        raise TypeError(f"Invalid schedule type: {scheduler_name}")

    return scheduler_1


def _configure_scheduler_2(optimizer, steps, warmup_steps_ratio, scheduler_name):
    # Configure the second part of the learning rate schedule
    if scheduler_name == "linear":
        # configure second lr scheduler
        scheduler_2 = LinearLR(
            optimizer,
            start_factor=1,
            end_factor=1e-4,
            total_iters=int(steps - warmup_steps_ratio * steps),
        )
    elif scheduler_name == "cosine":
        # configure second lr scheduler
        scheduler_2 = CosineAnnealingLR(
            optimizer, T_max=int(steps - warmup_steps_ratio * steps)
        )
    elif scheduler_name == "constant":
        # configure second lr scheduler
        scheduler_2 = ConstantLR(
            optimizer,
            factor=1,
        )

    return scheduler_2
