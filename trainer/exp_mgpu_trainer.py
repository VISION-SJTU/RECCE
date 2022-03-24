import os
import sys
import time
import math
import yaml
import torch
import random
import numpy as np

from tqdm import tqdm
from pprint import pprint
from torch.utils import data
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from dataset import load_dataset
from loss import get_loss
from model import load_model
from optimizer import get_optimizer
from scheduler import get_scheduler
from trainer import AbstractTrainer, LEGAL_METRIC
from trainer.utils import exp_recons_loss, MLLoss, reduce_tensor, center_print
from trainer.utils import MODELS_PATH, AccMeter, AUCMeter, AverageMeter, Logger, Timer


class ExpMultiGpuTrainer(AbstractTrainer):
    def __init__(self, config, stage="Train"):
        super(ExpMultiGpuTrainer, self).__init__(config, stage)
        np.random.seed(2021)

    def _mprint(self, content=""):
        if self.local_rank == 0:
            print(content)

    def _initiated_settings(self, model_cfg=None, data_cfg=None, config_cfg=None):
        self.local_rank = config_cfg["local_rank"]

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        # debug mode: no log dir, no train_val operation.
        self.debug = config_cfg["debug"]
        self._mprint(f"Using debug mode: {self.debug}.")
        self._mprint("*" * 20)

        self.eval_metric = config_cfg["metric"]
        if self.eval_metric not in LEGAL_METRIC:
            raise ValueError(f"Evaluation metric must be in {LEGAL_METRIC}, but found "
                             f"{self.eval_metric}.")
        if self.eval_metric == LEGAL_METRIC[-1]:
            self.best_metric = 1.0e8

        # distribution
        dist.init_process_group(config_cfg["distribute"]["backend"])

        # load training dataset
        train_dataset = data_cfg["file"]
        branch = data_cfg["train_branch"]
        name = data_cfg["name"]
        with open(train_dataset, "r") as f:
            options = yaml.load(f, Loader=yaml.FullLoader)
        train_options = options[branch]
        self.train_set = load_dataset(name)(train_options)
        # define training sampler
        self.train_sampler = data.distributed.DistributedSampler(self.train_set)
        # wrapped with data loader
        self.train_loader = data.DataLoader(self.train_set, shuffle=False,
                                            sampler=self.train_sampler,
                                            num_workers=data_cfg.get("num_workers", 4),
                                            batch_size=data_cfg["train_batch_size"])

        if self.local_rank == 0:
            # load validation dataset
            val_options = options[data_cfg["val_branch"]]
            self.val_set = load_dataset(name)(val_options)
            # wrapped with data loader
            self.val_loader = data.DataLoader(self.val_set, shuffle=True,
                                              num_workers=data_cfg.get("num_workers", 4),
                                              batch_size=data_cfg["val_batch_size"])

        self.resume = config_cfg.get("resume", False)

        if not self.debug:
            time_format = "%Y-%m-%d...%H.%M.%S"
            run_id = time.strftime(time_format, time.localtime(time.time()))
            self.run_id = config_cfg.get("id", run_id)
            self.dir = os.path.join("runs", self.model_name, self.run_id)

            if self.local_rank == 0:
                if not self.resume:
                    if os.path.exists(self.dir):
                        raise ValueError("Error: given id '%s' already exists." % self.run_id)
                    os.makedirs(self.dir, exist_ok=True)
                    print(f"Writing config file to file directory: {self.dir}.")
                    yaml.dump({"config": self.config,
                               "train_data": train_options,
                               "val_data": val_options},
                              open(os.path.join(self.dir, 'train_config.yml'), 'w'))
                    # copy the script for the training model
                    model_file = MODELS_PATH[self.model_name]
                    os.system("cp " + model_file + " " + self.dir)
                else:
                    print(f"Resuming the history in file directory: {self.dir}.")

                print(f"Logging directory: {self.dir}.")

                # redirect the std out stream
                sys.stdout = Logger(os.path.join(self.dir, 'records.txt'))
                center_print('Train configurations begins.')
                pprint(self.config)
                pprint(train_options)
                pprint(val_options)
                center_print('Train configurations ends.')

        # load model
        self.num_classes = model_cfg["num_classes"]
        self.device = "cuda:" + str(self.local_rank)
        self.model = load_model(self.model_name)(**model_cfg)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
        self._mprint(f"Using SyncBatchNorm.")
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        # load optimizer
        optim_cfg = config_cfg.get("optimizer", None)
        optim_name = optim_cfg.pop("name")
        self.optimizer = get_optimizer(optim_name)(self.model.parameters(), **optim_cfg)
        # load scheduler
        self.scheduler = get_scheduler(self.optimizer, config_cfg.get("scheduler", None))
        # load loss
        self.loss_criterion = get_loss(config_cfg.get("loss", None), device=self.device)

        # total number of steps (or epoch) to train
        self.num_steps = train_options["num_steps"]
        self.num_epoch = math.ceil(self.num_steps / len(self.train_loader))

        # the number of steps to write down a log
        self.log_steps = train_options["log_steps"]
        # the number of steps to validate on val dataset once
        self.val_steps = train_options["val_steps"]

        # balance coefficients
        self.lambda_1 = config_cfg["lambda_1"]
        self.lambda_2 = config_cfg["lambda_2"]
        self.warmup_step = config_cfg.get('warmup_step', 0)

        self.contra_loss = MLLoss()
        self.acc_meter = AccMeter()
        self.loss_meter = AverageMeter()
        self.recons_loss_meter = AverageMeter()
        self.contra_loss_meter = AverageMeter()

        if self.resume and self.local_rank == 0:
            self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _load_ckpt(self, best=False, train=False):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _save_ckpt(self, step, best=False):
        save_dir = os.path.join(self.dir, f"best_model_{step}.bin" if best else "latest_model.bin")
        torch.save({
            "step": step,
            "best_step": self.best_step,
            "best_metric": self.best_metric,
            "eval_metric": self.eval_metric,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, save_dir)

    def train(self):
        try:
            timer = Timer()
            grad_scalar = GradScaler(2 ** 10)
            if self.local_rank == 0:
                writer = None if self.debug else SummaryWriter(log_dir=self.dir)
                center_print("Training begins......")
            else:
                writer = None
            start_epoch = self.start_step // len(self.train_loader) + 1
            for epoch_idx in range(start_epoch, self.num_epoch + 1):
                # set sampler
                self.train_sampler.set_epoch(epoch_idx)

                # reset meter
                self.acc_meter.reset()
                self.loss_meter.reset()
                self.recons_loss_meter.reset()
                self.contra_loss_meter.reset()
                self.optimizer.step()

                train_generator = enumerate(self.train_loader, 1)
                # wrap train generator with tqdm for process 0
                if self.local_rank == 0:
                    train_generator = tqdm(train_generator, position=0, leave=True)

                for batch_idx, train_data in train_generator:
                    global_step = (epoch_idx - 1) * len(self.train_loader) + batch_idx
                    self.model.train()
                    I, Y = train_data
                    I = self.train_loader.dataset.load_item(I)
                    in_I, Y = self.to_device((I, Y))

                    # warm-up lr
                    if self.warmup_step != 0 and global_step <= self.warmup_step:
                        lr = self.config['config']['optimizer']['lr'] * float(global_step) / self.warmup_step
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr

                    self.optimizer.zero_grad()
                    with autocast():
                        Y_pre = self.model(in_I)

                        # for BCE Setting:
                        if self.num_classes == 1:
                            Y_pre = Y_pre.squeeze()
                            loss = self.loss_criterion(Y_pre, Y.float())
                            Y_pre = torch.sigmoid(Y_pre)
                        else:
                            loss = self.loss_criterion(Y_pre, Y)

                        # flood
                        loss = (loss - 0.04).abs() + 0.04
                        recons_loss = exp_recons_loss(self.model.module.loss_inputs['recons'], (in_I, Y))
                        contra_loss = self.contra_loss(self.model.module.loss_inputs['contra'], Y)
                        loss += self.lambda_1 * recons_loss + self.lambda_2 * contra_loss

                    grad_scalar.scale(loss).backward()
                    grad_scalar.step(self.optimizer)
                    grad_scalar.update()
                    if self.warmup_step == 0 or global_step > self.warmup_step:
                        self.scheduler.step()

                    self.acc_meter.update(Y_pre, Y, self.num_classes == 1)
                    self.loss_meter.update(reduce_tensor(loss).item())
                    self.recons_loss_meter.update(reduce_tensor(recons_loss).item())
                    self.contra_loss_meter.update(reduce_tensor(contra_loss).item())
                    iter_acc = reduce_tensor(self.acc_meter.mean_acc()).item()

                    if self.local_rank == 0:
                        if global_step % self.log_steps == 0 and writer is not None:
                            writer.add_scalar("train/Acc", iter_acc, global_step)
                            writer.add_scalar("train/Loss", self.loss_meter.avg, global_step)
                            writer.add_scalar("train/Recons_Loss",
                                              self.recons_loss_meter.avg if self.lambda_1 != 0 else 0.,
                                              global_step)
                            writer.add_scalar("train/Contra_Loss", self.contra_loss_meter.avg, global_step)
                            writer.add_scalar("train/LR", self.scheduler.get_last_lr()[0], global_step)

                        # log training step
                        train_generator.set_description(
                            "Train Epoch %d (%d/%d), Global Step %d, Loss %.4f, Recons %.4f, con %.4f, "
                            "ACC %.4f, LR %.6f" % (
                                epoch_idx, batch_idx, len(self.train_loader), global_step,
                                self.loss_meter.avg, self.recons_loss_meter.avg, self.contra_loss_meter.avg,
                                iter_acc, self.scheduler.get_last_lr()[0])
                        )

                        # validating process
                        if global_step % self.val_steps == 0 and not self.debug:
                            print()
                            self.validate(epoch_idx, global_step, timer, writer)

                    # when num_steps has been set and the training process will
                    # be stopped earlier than the specified num_epochs, then stop.
                    if self.num_steps is not None and global_step == self.num_steps:
                        if writer is not None:
                            writer.close()
                        if self.local_rank == 0:
                            print()
                            center_print("Training process ends.")
                        dist.destroy_process_group()
                        return
                # close the tqdm bar when one epoch ends
                if self.local_rank == 0:
                    train_generator.close()
                    print()
            # training ends with integer epochs
            if self.local_rank == 0:
                if writer is not None:
                    writer.close()
                center_print("Training process ends.")
            dist.destroy_process_group()
        except Exception as e:
            dist.destroy_process_group()
            raise e

    def validate(self, epoch, step, timer, writer):
        v_idx = random.randint(1, len(self.val_loader) + 1)
        categories = self.val_loader.dataset.categories
        self.model.eval()
        with torch.no_grad():
            acc = AccMeter()
            auc = AUCMeter()
            loss_meter = AverageMeter()
            cur_acc = 0.0  # Higher is better
            cur_auc = 0.0  # Higher is better
            cur_loss = 1e8  # Lower is better
            val_generator = tqdm(enumerate(self.val_loader, 1), position=0, leave=True)
            for val_idx, val_data in val_generator:
                I, Y = val_data
                I = self.val_loader.dataset.load_item(I)
                in_I, Y = self.to_device((I, Y))
                Y_pre = self.model(in_I)

                # for BCE Setting:
                if self.num_classes == 1:
                    Y_pre = Y_pre.squeeze()
                    loss = self.loss_criterion(Y_pre, Y.float())
                    Y_pre = torch.sigmoid(Y_pre)
                else:
                    loss = self.loss_criterion(Y_pre, Y)

                acc.update(Y_pre, Y, self.num_classes == 1)
                auc.update(Y_pre, Y, self.num_classes == 1)
                loss_meter.update(loss.item())

                cur_acc = acc.mean_acc()
                cur_loss = loss_meter.avg

                val_generator.set_description(
                    "Eval Epoch %d (%d/%d), Global Step %d, Loss %.4f, ACC %.4f" % (
                        epoch, val_idx, len(self.val_loader), step,
                        cur_loss, cur_acc)
                )

                if val_idx == v_idx or val_idx == 1:
                    sample_recons = list()
                    for _ in self.model.module.loss_inputs['recons']:
                        sample_recons.append(_[:4].to("cpu"))
                    # show images
                    images = I[:4]
                    images = torch.cat([images, *sample_recons], dim=0)
                    pred = Y_pre[:4]
                    gt = Y[:4]
                    figure = self.plot_figure(images, pred, gt, 4, categories, show=False)

            cur_auc = auc.mean_auc()
            print("Eval Epoch %d, Loss %.4f, ACC %.4f, AUC %.4f" % (epoch, cur_loss, cur_acc, cur_auc))
            if writer is not None:
                writer.add_scalar("val/Loss", cur_loss, step)
                writer.add_scalar("val/Acc", cur_acc, step)
                writer.add_scalar("val/AUC", cur_auc, step)
                writer.add_figure("val/Figures", figure, step)
            # record the best acc and the corresponding step
            if self.eval_metric == 'Acc' and cur_acc >= self.best_metric:
                self.best_metric = cur_acc
                self.best_step = step
                self._save_ckpt(step, best=True)
            elif self.eval_metric == 'AUC' and cur_auc >= self.best_metric:
                self.best_metric = cur_auc
                self.best_step = step
                self._save_ckpt(step, best=True)
            elif self.eval_metric == 'LogLoss' and cur_loss <= self.best_metric:
                self.best_metric = cur_loss
                self.best_step = step
                self._save_ckpt(step, best=True)
            print("Best Step %d, Best %s %.4f, Running Time: %s, Estimated Time: %s" % (
                self.best_step, self.eval_metric, self.best_metric,
                timer.measure(), timer.measure(step / self.num_steps)
            ))
            self._save_ckpt(step, best=False)

    def test(self):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")
