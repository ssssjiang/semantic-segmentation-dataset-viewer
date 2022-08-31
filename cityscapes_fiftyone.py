# https://github.com/haritsahm/pytorch-DMANet/blob/main/notebooks/dataset-preparation.ipynb

import os
import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.zoo as foz

classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
class_map = dict(zip(range(19), classes))

DATASET_DIR = "/home/songshu/dataset/Foggy_Zurich/images/"
FO_SEGMENTATION_DIR = "/home/songshu/dataset/Foggy_Zurich/cityscape_fo_image-segmentation"
FO_COCO_DIR = "/home/songshu/dataset/Foggy_Zurich/cityscape_fo_coco-detection"

for split in ["train", "validation", "test"]:
    dataset = foz.load_zoo_dataset(
        "cityscapes",
        split=split,
        source_dir=DATASET_DIR,
        dataset_dir=os.path.join(DATASET_DIR, "fiftyone_cityscape"),
    )

    match = F("label").is_in(classes)
    if split != "test":
        matching_view = dataset.match(
            F("gt_fine.polylines").filter(match).length() > 0
        )
    else:
        matching_view = dataset

    # Generate ImageSegmentationDirectory format
    matching_view.export(
        dataset_type=fo.types.ImageSegmentationDirectory,
        export_dir=FO_SEGMENTATION_DIR,
        data_path=f"data_{split}/",
        labels_path=f"labels_{split}/",
        label_field="gt_fine",
        export_media="symlink",
        mask_targets=class_map)

    # Generate COCODetectionDataset format
    matching_view.export(
        export_dir=FO_COCO_DIR,
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=f"labels/{split}.json",
        label_field="gt_fine",
        export_media="symlink",
        classes=classes,
    )

session = fo.launch_app(dataset)
session.view = dataset.take(100)