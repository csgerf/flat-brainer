import os
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.core.logging.meter import AverageValueMeter
from src.initialization import build_trainer_from_config


scaler = GradScaler()
device = "cuda"


def train_step(model, optimizer, loss_fn, batch, enable_autocast=False):
    data = batch["image"].to(device)
    target = batch["label"].to(device)
    optimizer.zero_grad()

    # Runs the forward pass with autocasting.
    with autocast(enabled=enable_autocast):
        # logits, output = model(data.float())
        logits = model(data)
        loss = loss_fn(logits, target.float())

    if enable_autocast:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # preds = output.detach().cpu()
    preds = torch.sigmoid(logits).detach().cpu()
    # preds = output.detach().cpu()
    return loss.item(), preds


def validate_batch(model, loss_fn, batch):
    data = batch["image"].to(device)
    target = batch["label"].to(device)

    with torch.no_grad():
        # logits, output = model(data.float())
        logits = model(data.float())
        loss = loss_fn(logits, target.float())
        # loss = loss_fn(logits, target)
        # preds = output.detach().cpu()

    preds = torch.sigmoid(logits).detach().cpu()

    return loss.item(), preds


def run_single_epoch(model, optimizer, scheduler, loss_fn, data_loader, postfix_dict, hooks=None, epoch=-1, mode="train"):
    num_steps = len(data_loader)
    iterator = tqdm(
        enumerate(data_loader), total=num_steps, ncols=0, position=0, leave=True
    )
    loss_meter = AverageValueMeter()
    optimizer.zero_grad()

    for i, batch in iterator:
        if mode == 'train':
            loss, output = train_step(model, optimizer, loss_fn, batch)
        else:
            loss, output = validate_batch(model, loss_fn, batch)

        loss_meter.add(loss)
        if i % 5 == 0:
            ex.log_scalar("{}/loss".format(mode), loss, epoch + i / num_steps)
            postfix_dict["{}/loss".format(mode)] = "{:.4f}".format(loss_meter.get_mean())

            f_epoch = epoch + i / num_steps
            desc = "{:5s}".format("{:s}".format(mode))
            desc += ", {:04d}/{:04d}, {:.2f} epoch".format(i, num_steps, f_epoch)
            iterator.set_description(desc)
            iterator.set_postfix(**postfix_dict)

    return postfix_dict


def train(config, model, optimizer, scheduler, loss_fn, dataloaders, hooks=None, last_epoch=-1):
    trainer_config = config.trainer

    writer = SummaryWriter(log_dir=trainer_config.output_dir)

    num_epochs = trainer_config.params.num_epochs
    model.to(device)

    postfix_dict = {}
    train_loader = dataloaders["train"]
    valid_loader = dataloaders["val"]
    for epoch in range(last_epoch, num_epochs + 1):
        model.train()
        postfix_dict = run_single_epoch(data_loader=train_loader,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        loss_fn=loss_fn,
                                        epoch=epoch,
                                        postfix_dict=postfix_dict,
                                        mode='train')
        scheduler.step()

        model.eval()
        postfix_dict = run_single_epoch(data_loader=valid_loader,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        loss_fn=loss_fn,
                                        epoch=epoch,
                                        postfix_dict=postfix_dict,
                                        mode='val')


def run(config):
    model, optimizer, scheduler, loss_fn, data_loaders = build_trainer_from_config(config)

    cfg = OmegaConf.to_container(config)
    cfg["optimizer"]["params"] = optimizer.defaults
    cfg["scheduler"]["params"] = scheduler.state_dict()
    cfg = OmegaConf.create(cfg)
    output_dir = config.trainer.get("output_dir", "runs")
    save_config_path = os.path.join(output_dir, "experiment_config.yaml")
    OmegaConf.save(cfg, save_config_path)

    train(config, model, optimizer, scheduler, loss_fn, data_loaders)
