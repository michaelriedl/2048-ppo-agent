import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import torch

from src.optim.configure_optimizers import configure_bert_optimizers


def test_call():
    # Create a dummy model
    model = torch.nn.Linear(10, 10)
    if torch.cuda.is_available():
        model = model.to("cuda")
    # Set the parameters
    opt_name = "adamw"
    max_lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.01
    warmup_steps_ratio = 0.1
    steps = 1000
    scheduler_names = ["linear", "constant"]
    blacklist_weight_modules = []
    # Call the function
    opt_dict = configure_bert_optimizers(
        model,
        opt_name,
        max_lr,
        betas,
        eps,
        weight_decay,
        warmup_steps_ratio,
        steps,
        scheduler_names,
        blacklist_weight_modules,
    )

    assert isinstance(opt_dict, dict)
