import yaml
import argparse

from trainer import ExpMultiGpuTrainer


def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        type=str,
                        default="config/Recce.yml",
                        help="Specified the path of configuration file to be used.")
    parser.add_argument("--local_rank", default=0,
                        type=int,
                        help="Specified the node rank for distributed training.")
    return parser.parse_args()


if __name__ == '__main__':
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    arg = arg_parser()
    config = arg.config

    with open(config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config["config"]["local_rank"] = arg.local_rank

    trainer = ExpMultiGpuTrainer(config, stage="Train")
    trainer.train()
