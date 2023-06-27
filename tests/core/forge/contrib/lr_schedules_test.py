import torch.nn as nn
from src.core.forge.contrib.lr_schedules import *

if __name__ == "__main__":
    import matplotlib as mpl

    mpl.use("module://backend_interagg")
    import matplotlib.pyplot as plt

    from torch.optim import SGD, Optimizer

    net = nn.Conv2d(1, 1, 1)
    opt = SGD(net.parameters(), lr=1e-3)

    epochs = 100

    plt.figure()

    scheduler = OnceCycleLR(opt, epochs + 1, min_lr_factor=0.01)
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    plt.plot(range(epochs), lrs, label="1cycle")

    scheduler = CosineAnnealingLRWithDecay(opt, round(epochs / 5), gamma=0.99)
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    plt.plot(range(epochs), lrs, label="cosine")

    scheduler = PolyLR(opt, epochs, gamma=0.9)
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    plt.plot(range(epochs), lrs, label="poly")

    scheduler = FlatCosineAnnealingLR(opt, epochs, T_flat=epochs // 2, eta_min=1e-6)
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    plt.plot(range(epochs), lrs, label="flat_cos")

    plt.legend()
    plt.show()
