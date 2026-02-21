from pathlib import Path

BASE_UA = Path(r"D:\Learn\Year4\KLTN\Dataset\UA-DETRAC\DETRAC_Upload")
BASE_COCO = Path(r"D:\Learn\Year4\KLTN\Dataset\COCO2017")
OUT_ROOT = Path(r"D:/Datasets/mixed_dataset")

paths = {
    'COCO_TRAIN_IMG': BASE_COCO / 'train2017',
    'COCO_ANN_TRAIN': BASE_COCO / 'annotations' / 'instances_train2017.json',
    'UA_TRAIN_IMG': BASE_UA / 'images' / 'train',
    'UA_VAL_IMG': BASE_UA / 'images' / 'val',
    'UA_TRAIN_LABEL': BASE_UA / 'labels' / 'train',
    'UA_VAL_LABEL': BASE_UA / 'labels' / 'val',
    'OUT_ROOT': OUT_ROOT
}

print('Checking dataset paths:')
for name, p in paths.items():
    print(f"- {name}: {p} -> exists: {p.exists()}")

# Also list a few sample files if directories exist
if paths['COCO_TRAIN_IMG'].exists():
    imgs = list(paths['COCO_TRAIN_IMG'].glob('*.jpg'))[:5]
    print('\nSample COCO images:')
    for im in imgs:
        print('  ', im.name)

if paths['UA_TRAIN_LABEL'].exists():
    lbls = list(paths['UA_TRAIN_LABEL'].glob('*.txt'))[:5]
    print('\nSample UA train labels:')
    for l in lbls:
        print('  ', l.name)
