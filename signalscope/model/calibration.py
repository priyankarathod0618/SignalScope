"""
calibration.py — temperature scaling so confidence scores are honest
("not sure" should genuinely mean not sure). Fit on the validation split
only, never on the organizers' held-out set.
"""

import torch
import torch.nn as nn


def fit_temperature(model, val_loader, device, max_iter=50, lr=0.01) -> float:
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits_list.append(model(x).cpu())
            labels_list.append(y)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = logits / temperature.clamp(min=0.05)
        loss = nll(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.clamp(min=0.05).item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 0.05)
