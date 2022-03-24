import os
import sys
import yaml
import torch
import random

from tqdm import tqdm
from pprint import pprint
from torch.utils import data

from dataset import load_dataset
from loss import get_loss
from model import load_model
from model.common import freeze_weights
from trainer import AbstractTrainer
from trainer.utils import AccMeter, AUCMeter, AverageMeter, Logger, center_print


class ExpTester(AbstractTrainer):
    def __init__(self, config, stage="Test"):
        super(ExpTester, self).__init__(config, stage)

        if torch.cuda.is_available() and self.device is not None:
            print(f"Using cuda device: {self.device}.")
            self.gpu = True
            self.model = self.model.to(self.device)
        else:
            print("Using cpu device.")
            self.device = torch.device("cpu")

    def _initiated_settings(self, model_cfg=None, data_cfg=None, config_cfg=None):
        self.gpu = False
        self.device = config_cfg.get("device", None)

    def _train_settings(self, model_cfg=None, data_cfg=None, config_cfg=None):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _test_settings(self, model_cfg=None, data_cfg=None, config_cfg=None):
        # load test dataset
        test_dataset = data_cfg["file"]
        branch = data_cfg["test_branch"]
        name = data_cfg["name"]
        with open(test_dataset, "r") as f:
            options = yaml.load(f, Loader=yaml.FullLoader)
        test_options = options[branch]
        self.test_set = load_dataset(name)(test_options)
        # wrapped with data loader
        self.test_batch_size = data_cfg["test_batch_size"]
        self.test_loader = data.DataLoader(self.test_set, shuffle=False,
                                           batch_size=self.test_batch_size)
        self.run_id = config_cfg["id"]
        self.ckpt_fold = config_cfg.get("ckpt_fold", "runs")
        self.dir = os.path.join(self.ckpt_fold, self.model_name, self.run_id)

        # load model
        self.num_classes = model_cfg["num_classes"]
        self.model = load_model(self.model_name)(**model_cfg)

        # load loss
        self.loss_criterion = get_loss(config_cfg.get("loss", None))

        # redirect the std out stream
        sys.stdout = Logger(os.path.join(self.dir, "test_result.txt"))
        print('Run dir: {}'.format(self.dir))

        center_print('Test configurations begins')
        pprint(self.config)
        pprint(test_options)
        center_print('Test configurations ends')

        self.ckpt = config_cfg.get("ckpt", "best_model")
        self._load_ckpt(best=True, train=False)

    def _save_ckpt(self, step, best=False):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _load_ckpt(self, best=False, train=False):
        load_dir = os.path.join(self.dir, self.ckpt + ".bin" if best else "latest_model.bin")
        load_dict = torch.load(load_dir, map_location=self.device)
        self.start_step = load_dict["step"]
        self.best_step = load_dict["best_step"]
        self.best_metric = load_dict.get("best_metric", None)
        if self.best_metric is None:
            self.best_metric = load_dict.get("best_acc")
        self.eval_metric = load_dict.get("eval_metric", None)
        if self.eval_metric is None:
            self.eval_metric = load_dict.get("Acc")
        self.model.load_state_dict(load_dict["model"])
        print(f"Loading checkpoint from {load_dir}, best step: {self.best_step}, "
              f"best {self.eval_metric}: {round(self.best_metric.item(), 4)}.")

    def train(self):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def validate(self, epoch, step, timer, writer):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def test(self, display_images=False):
        freeze_weights(self.model)
        t_idx = random.randint(1, len(self.test_loader) + 1)
        self.fixed_randomness()  # for reproduction

        acc = AccMeter()
        auc = AUCMeter()
        logloss = AverageMeter()
        test_generator = tqdm(enumerate(self.test_loader, 1))
        categories = self.test_loader.dataset.categories
        for idx, test_data in test_generator:
            self.model.eval()
            I, Y = test_data
            I = self.test_loader.dataset.load_item(I)
            if self.gpu:
                in_I, Y = self.to_device((I, Y))
            else:
                in_I, Y = (I, Y)
            Y_pre = self.model(in_I)

            # for BCE Setting:
            if self.num_classes == 1:
                Y_pre = Y_pre.squeeze()
                loss = self.loss_criterion(Y_pre, Y.float())
                Y_pre = torch.sigmoid(Y_pre)
            else:
                loss = self.loss_criterion(Y_pre, Y)

            acc.update(Y_pre, Y, use_bce=self.num_classes == 1)
            auc.update(Y_pre, Y, use_bce=self.num_classes == 1)
            logloss.update(loss.item())

            test_generator.set_description("Test %d/%d" % (idx, len(self.test_loader)))
            if display_images and idx == t_idx:
                # show images
                images = I[:4]
                pred = Y_pre[:4]
                gt = Y[:4]
                self.plot_figure(images, pred, gt, 2, categories)

        print("Test, FINAL LOSS %.4f, FINAL ACC %.4f, FINAL AUC %.4f" %
              (logloss.avg, acc.mean_acc(), auc.mean_auc()))
        auc.curve(self.dir)
