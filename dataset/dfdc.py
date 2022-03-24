import json
from glob import glob
from os.path import join
from dataset import AbstractDataset

SPLIT = ["train", "val", "test"]
LABEL_MAP = {"REAL": 0, "FAKE": 1}


class DFDC(AbstractDataset):
    """
    Deepfake Detection Challenge organized by Facebook
    """

    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if cfg['split'] not in SPLIT:
            raise ValueError(f"split should be one of {SPLIT}, but found {cfg['split']}.")
        super(DFDC, self).__init__(cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'DFDC' of split '{cfg['split']}'"
              f"\nPlease wait patiently...")
        self.categories = ['original', 'fake']
        self.root = cfg['root']
        self.num_real = 0
        self.num_fake = 0
        if self.split == "test":
            self.__load_test_data()
        elif self.split == "train":
            self.__load_train_data()
        assert len(self.images) == len(self.targets), "Length of images and targets not the same!"
        print(f"Data from 'DFDC' loaded.")
        print(f"Real: {self.num_real}, Fake: {self.num_fake}.")
        print(f"Dataset contains {len(self.images)} images\n")

    def __load_test_data(self):
        label_path = join(self.root, "test", "labels.csv")
        with open(label_path, encoding="utf-8") as file:
            content = file.readlines()
        for _ in content:
            if ".mp4" in _:
                key = _.split(".")[0]
                label = _.split(",")[1].strip()
                label = int(label)
                imgs = glob(join(self.root, "test", "images", key, "*.png"))
                num = len(imgs)
                self.images.extend(imgs)
                self.targets.extend([label] * num)
                if label == 0:
                    self.num_real += num
                elif label == 1:
                    self.num_fake += num

    def __load_train_data(self):
        train_folds = glob(join(self.root, "dfdc_train_part_*"))
        for fold in train_folds:
            fold_imgs = list()
            fold_tgts = list()
            metadata_path = join(fold, "metadata.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    metadata = json.loads(file.readline())
                for k, v in metadata.items():
                    index = k.split(".")[0]
                    label = LABEL_MAP[v["label"]]
                    imgs = glob(join(fold, "images", index, "*.png"))
                    fold_imgs.extend(imgs)
                    fold_tgts.extend([label] * len(imgs))
                    if label == 0:
                        self.num_real += len(imgs)
                    elif label == 1:
                        self.num_fake += len(imgs)
                self.images.extend(fold_imgs)
                self.targets.extend(fold_tgts)
            except FileNotFoundError:
                continue


if __name__ == '__main__':
    import yaml

    config_path = "../config/dataset/dfdc.yml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config["train_cfg"]
    # config = config["test_cfg"]


    def run_dataset():
        dataset = DFDC(config)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataset):
            path, target = _
            print(f"path: {path}, target: {target}")
            if i >= 9:
                break


    def run_dataloader(display_samples=False):
        from torch.utils import data
        import matplotlib.pyplot as plt

        dataset = DFDC(config)
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
