from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F


class SSMixupBatch(NamedTuple):
    inputs: Tensor
    targets: Tensor
    psi: Tensor
    mask: Tensor


def build_ssmixup_inputs(
    inputs: Tensor,
    pseudo_labels: Tensor,
    confidences: Tensor,
    threshold: float,
    eps: float,
    device: torch.device,
    num_classes: int,
) -> SSMixupBatch:
    batch = inputs.size(0)
    perm = torch.randperm(batch, device=device)
    psi = confidences / (confidences + confidences[perm] + eps)
    psi_input = psi.view(-1, *([1] * (inputs.dim() - 1)))
    mixed_inputs = psi_input * inputs + (1.0 - psi_input) * inputs[perm]

    soft_labels = F.one_hot(pseudo_labels, num_classes=num_classes).float()
    mixed_targets = psi.unsqueeze(1) * soft_labels + (1.0 - psi).unsqueeze(1) * soft_labels[perm]

    mask = confidences.ge(threshold).float() * confidences[perm].ge(threshold).float()

    return SSMixupBatch(
        inputs=mixed_inputs,
        targets=mixed_targets,
        psi=psi,
        mask=mask,
    )
