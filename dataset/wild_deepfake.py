import torch
import numpy as np
from os.path import join
from dataset import AbstractDataset

SPLITS = ["train", "test"]


class WildDeepfake(AbstractDataset):
    """
    Wild Deepfake Dataset proposed in "WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection"
    """

    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if cfg['split'] not in SPLITS:
            raise ValueError(f"split should be one of {SPLITS}, but found {cfg['split']}.")
        super(WildDeepfake, self).__init__(cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'WildDeepfake' of split '{cfg['split']}'"
              f"\nPlease wait patiently...")
        self.categories = ['original', 'fake']
        self.root = cfg['root']
        self.num_train = cfg.get('num_image_train', None)
        self.num_test = cfg.get('num_image_test', None)
        self.images, self.targets = self.__get_images()
        print(f"Data from 'WildDeepfake' loaded.")
        print(f"Dataset contains {len(self.images)} images.\n")

    def __get_images(self):
        if self.split == 'train':
            num = self.num_train
        elif self.split == 'test':
            num = self.num_test
        else:
            num = None
        real_images = torch.load(join(self.root, self.split, "real.pickle"))
        if num is not None:
            real_images = np.random.choice(real_images, num // 3, replace=False)
        real_tgts = [torch.tensor(0)] * len(real_images)
        print(f"real: {len(real_tgts)}")
        fake_images = torch.load(join(self.root, self.split, "fake.pickle"))
        if num is not None:
            fake_images = np.random.choice(fake_images, num - num // 3, replace=False)
        fake_tgts = [torch.tensor(1)] * len(fake_images)
        print(f"fake: {len(fake_tgts)}")
        return real_images + fake_images, real_tgts + fake_tgts

    def __getitem__(self, index):
        path = join(self.root, self.split, self.images[index])
        tgt = self.targets[index]
        return path, tgt


if __name__ == '__main__':
    import yaml

    config_path = "../config/dataset/wilddeepfake.yml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config["train_cfg"]
    # config = config["test_cfg"]


    def run_dataset():
        dataset = WildDeepfake(config)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataset):
            path, target = _
            print(f"path: {path}, target: {target}")
            if i >= 9:
                break


    def run_dataloader(display_samples=False):
        from torch.utils import data
        import matplotlib.pyplot as plt

        dataset = WildDeepfake(config)
        dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataloader):
            path, targets = _
            image = dataloader.dataset.load_item(path)
            print(f"image: {image.shape}, target: {targets}")
            if display_samples:
                plt.figure()
                img = image[0].permute([1, 2, 0]).numpy()
                plt.imshow(img)
                # plt.savefig("./img_" + str(i) + ".png")
                plt.show()
            if i >= 9:
                break


    ###########################
    # run the functions below #
    ###########################

    # run_dataset()
    run_dataloader(False)
