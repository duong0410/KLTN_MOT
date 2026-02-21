import os
import json
import random
import shutil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from PIL import Image

# CONFIG — CHANGE THESE 3 PATHS
BASE_UA = Path(r"D:\Learn\Year4\KLTN\Dataset\UA-DETRAC\DETRAC_Upload")
BASE_COCO = Path(r"D:/Datasets/COCO")
OUT_ROOT = Path(r"D:/Datasets/mixed_dataset")

UA_TRAIN_IMG = BASE_UA/"images/train"
UA_VAL_IMG   = BASE_UA/"images/val"
UA_TRAIN_LABEL = BASE_UA/"labels/train"
UA_VAL_LABEL   = BASE_UA/"labels/val"

COCO_TRAIN_IMG = BASE_COCO/"train2017"
COCO_ANN_TRAIN = BASE_COCO/"annotations/instances_train2017.json"

# unified dataset out
TRAIN_OUT_IMG = OUT_ROOT/"train/images"
TRAIN_OUT_LABEL = OUT_ROOT/"train/labels"
VAL_OUT_IMG = OUT_ROOT/"val/images"
VAL_OUT_LABEL = OUT_ROOT/"val/labels"
for p in [TRAIN_OUT_IMG, TRAIN_OUT_LABEL, VAL_OUT_IMG, VAL_OUT_LABEL]:
    p.mkdir(parents=True, exist_ok=True)

# CLASSES
UNIFIED_CLASSES = ["car", "truck", "bus", "van", "motorcycle", "bicycle", "person"]
name_to_id = {n: i for i, n in enumerate(UNIFIED_CLASSES)}

UA_MAP = {
    "0": "truck",
    "1": "car",
    "2": "van",
    "3": "bus"
}

# CONFIG
COCO_TARGET_COUNT = 5000
UA_TRAIN_LIMIT = 15000
UA_VAL_LIMIT = 3000

# HELPERS
def coco_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    cx = x + bw/2
    cy = y + bh/2
    return cx/w, cy/h, bw/w, bh/h


def sample_coco_balanced(images, anns_per_image, target_count):
    class_to_imgs = defaultdict(list)
    for img in images:
        img_id = img["id"]
        cats = set(anns_per_image.get(img_id, []))
        for c in cats:
            class_to_imgs[c].append(img_id)

    n_class = len(class_to_imgs)
    per_class = max(1, target_count // n_class)

    selected = []
    for c, lst in class_to_imgs.items():
        if len(lst) >= per_class:
            selected += random.sample(lst, per_class)
        else:
            selected += lst + random.choices(lst, k=per_class - len(lst))

    selected = list(dict.fromkeys(selected))
    return set(selected[:target_count])

# LOAD COCO
print("(temp) Loading COCO annotations... (syntax-only check file)")
with open(COCO_ANN_TRAIN, "r") as f:
    coco = json.load(f)

coco_images = coco["images"]
coco_annotations = coco["annotations"]
coco_categories = coco["categories"]
id2name = {c["id"]: c["name"] for c in coco_categories}

coco_annotations = [ann for ann in coco_annotations if id2name[ann["category_id"]] in UNIFIED_CLASSES]

anns_per_img = defaultdict(list)
for ann in coco_annotations:
    anns_per_img[ann["image_id"]].append(ann["category_id"])

print("(temp) Sampling COCO...")
selected_ids = sample_coco_balanced(coco_images, anns_per_img, COCO_TARGET_COUNT)
selected_images = [img for img in coco_images if img["id"] in selected_ids]
selected_anns = [ann for ann in coco_annotations if ann["image_id"] in selected_ids]

imgid2info = {img["id"]: img for img in selected_images}

print(f"(temp) Selected {len(selected_images)} images from COCO")

# COPY COCO IMAGES + LABELS (no-op for syntax check)
for img in tqdm(selected_images, desc="Copy COCO"):
    src = COCO_TRAIN_IMG / img["file_name"]
    dst = TRAIN_OUT_IMG / img["file_name"]
    # don't actually copy in this temp file

anns_by_img = defaultdict(list)
for ann in selected_anns:
    anns_by_img[ann["image_id"]].append(ann)

for img_id, anns in tqdm(anns_by_img.items(), desc="Write COCO labels"):
    info = imgid2info[img_id]
    iw, ih = info["width"], info["height"]

    fname = info["file_name"]
    out = []
    for ann in anns:
        cname = id2name[ann["category_id"]]
        cid = name_to_id[cname]
        xc, yc, w, h = coco_to_yolo(ann["bbox"], iw, ih)
        out.append(f"{cid} {xc} {yc} {w} {h}")

    # no file write here

# LOAD & OVERSAMPLE UA-DETRAC

def compute_class_imgs(label_dir):
    class_to_imgs = defaultdict(list)
    for lbl in tqdm(list(label_dir.glob("*.txt")), desc=f"Scan {label_dir.name}"):
        stem = lbl.stem
        with open(lbl, "r") as f:
            seen = set()
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                ua = parts[0]
                if ua not in UA_MAP:
                    continue
                cname = UA_MAP[ua]
                if cname not in seen:
                    class_to_imgs[cname].append(stem)
                    seen.add(cname)
    return class_to_imgs


def oversample(class_to_imgs, target=None):
    counts = {c: len(v) for c, v in class_to_imgs.items()}
    max_count = max(counts.values())
    target = target or max_count

    result = []
    for c, lst in class_to_imgs.items():
        if len(lst) >= target:
            chosen = random.sample(lst, target)
        else:
            chosen = lst + random.choices(lst, k=target-len(lst))
        result.extend(chosen)
    random.shuffle(result)
    return result

train_imgs_map = compute_class_imgs(UA_TRAIN_LABEL)
val_imgs_map   = compute_class_imgs(UA_VAL_LABEL)

train_final = oversample(train_imgs_map)[:UA_TRAIN_LIMIT]
val_final   = oversample(val_imgs_map)[:UA_VAL_LIMIT]

print("(temp) UA final train:", len(train_final))
print("(temp) UA final val:", len(val_final))

# COPY UA-DETRAC (no-op)
def copy_ua(stems, src_img, src_lbl, dst_img, dst_lbl):
    for stem in tqdm(stems, desc=f"Copy {src_img.name}"):
        src_i = src_img / f"{stem}.jpg"
        if src_i.exists():
            pass

        lbl = src_lbl / f"{stem}.txt"
        out_lines = []
        if lbl.exists():
            with open(lbl, "r") as f:
                for line in f:
                    p = line.split()
                    if len(p) != 5:
                        continue
                    ua = p[0]
                    if ua not in UA_MAP:
                        continue
                    cname = UA_MAP[ua]
                    cid = name_to_id[cname]
                    out_lines.append(f"{cid} {p[1]} {p[2]} {p[3]} {p[4]}")

        # no file write here

print("(temp) Done building mixed dataset (syntax-only run)")

# Write YAML (no-op)
print("(temp) Dataset YAML path:", OUT_ROOT/"dataset_yaml.yaml")
