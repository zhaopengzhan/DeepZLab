import torch

from models.LSeg.models.lseg_net import LSegNet


class MyLSeg(torch.nn.Module):
    def __init__(self, crop_size=480):
        super().__init__()

        self.net = LSegNet(
            labels=self.get_labels(),
            backbone='clip_vitl16_384',
            features=256,
            crop_size=crop_size,
            arch_option=0,
            block_depth=0,
            activation='lrelu',
        )

        self.net.pretrained.model.patch_embed.img_size = (crop_size, crop_size)

        self.label_set = [
            'nodata', 'Open Water', 'Perennial Ice/Snow',
            # 2
            'Developed, Open Space', 'Developed, Low Intensity', 'Developed, Medium Intensity',
            'Developed, High Intensity',
            # 3
            'Barren Land (Rock/Sand/Clay)',
            # 4
            'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest',
            # 5 7
            'Shrub/Scrub', 'Grassland/Herbaceous',
            # 8
            'Pasture/Hay', 'Cultivated Crops',
            # 9
            'Woody Wetlands', 'Emergent Herbaceous Wetlands',
        ]

    def get_labels(self):
        return ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet',
                'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
                'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence',
                'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator',
                'grandstand', 'path', 'stairs', 'runway', 'case', 'pool', 'pillow', 'screen', 'stairway', 'river',
                'bridge', 'bookcase', 'blind', 'coffee', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop',
                'stove', 'palm', 'kitchen', 'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel', 'bus', 'towel',
                'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television', 'airplane',
                'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster',
                'stage', 'van', 'ship', 'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming', 'stool',
                'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step',
                'tank', 'trade', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                'sculpture', 'hood', 'sconce', 'vase', 'traffic', 'tray', 'ashcan', 'fan', 'pier', 'crt', 'plate',
                'monitor', 'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag']

    def forward(self, image, label_set=None):
        if label_set is None:
            label_set = self.label_set
        return self.net(image, label_set)
