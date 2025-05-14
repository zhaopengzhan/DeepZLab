import glob

from torch.utils.data import DataLoader

from dataloader.Poland import Poland


def build_dataloader(data_name):

    if data_name == 'L2HNetV2':
        return build_chesapeake()
    pass


def build_poland(args):
    args.batch_size = 8
    args.num_workers = 2
    args.num_chips_per_tile = 4
    args.image_size = 224
    args.stride = 112
    args.image_re_path = r'F:\datasets\OpenEarthMap\OpenEarthMap_wo_xBD\*\labels\*.tif'
    train_label_hr_list = glob.glob(args.image_re_path)
    train_label_lr_list = [filename.replace('labels', 'ESA_GLC10') for filename in train_label_hr_list]
    train_image_list = [filename.replace('labels', 'images') for filename in train_label_hr_list]

    train_ds = Poland(train_image_list, train_label_lr_list, train_label_hr_list, label_type=args.label_type,
                      chip_size=args.image_size, num_chips_per_tile=args.num_chips_per_tile)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=False)

    test_dl = None

    print('dataloader finish')
    return train_dl, test_dl



def build_chesapeake(args):
    args.batch_size = 8
    args.num_workers = 2
    args.num_chips_per_tile = 4
    args.image_size = 224
    args.stride = 112
    args.image_re_path = r'F:\datasets\OpenEarthMap\OpenEarthMap_wo_xBD\*\labels\*.tif'
    train_label_hr_list = glob.glob(args.image_re_path)
    train_label_lr_list = [filename.replace('labels', 'ESA_GLC10') for filename in train_label_hr_list]
    train_image_list = [filename.replace('labels', 'images') for filename in train_label_hr_list]

    train_ds = Poland(train_image_list, train_label_lr_list, train_label_hr_list, label_type=args.label_type,
                      chip_size=args.image_size, num_chips_per_tile=args.num_chips_per_tile)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=False)

    test_dl = None

    print('dataloader finish')
    return train_dl, test_dl
