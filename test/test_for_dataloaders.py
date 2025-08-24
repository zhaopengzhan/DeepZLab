import warnings

from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")


import dataloader


def test_list_dataloader():
    model_names = dataloader.list_dataloaders()
    print(model_names)
    pass


def test_build_all_dataloader(in_channels=4, image_size=224, num_classes=17):
    names = dataloader.list_dataloaders()
    for name in names:
        train_ds = dataloader.build_dataloader(name)
        train_dl = DataLoader(train_ds, batch_size=2, drop_last=True, num_workers=4, pin_memory=True)
        print(len(train_dl))
        pbar = tqdm(train_dl)
        for batch_idx, batch in enumerate(pbar):
            for i in range(len(batch)):
                print(f'batch {i}: {batch[i].shape}')

            break

    pass


# def test_for_deprecated():
#     dataloader.build_dataloader1(name='Chesapeake_L2H', image_list=[], label_lr_list=[], label_hr_list=[])


if __name__ == '__main__':
    # test_list_dataloader()
    # test_build_all_dataloader()
    # test_for_deprecated()
    pass
