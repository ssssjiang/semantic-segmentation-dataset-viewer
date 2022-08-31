import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


name = "my-dataset"
data_path = "/home/songshu/dataset/Foggy_Zurich/images/"
labels_path = "/home/songshu/dataset/Foggy_Zurich/gt_labelTrainIds/"
export_dir="/home/songshu/dataset/Foggy_Zurich"

classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
class_map = dict(zip(range(19), classes))

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.ImageSegmentationDirectory,
    data_path=data_path,
    labels_path=labels_path,
    name=name,
    tags=["ProjectA"]
)

dataset.default_mask_targets = class_map

dataset.mask_target = {
    "ground_truth": class_map
}


# for data in dataset:
#     data.add_labels(data.id, "gt")
#     print(data)
#
# dataset.save()

session = fo.launch_app(dataset, desktop=True)

# label_field = "ground_truth"  # for example
#
# # Export the dataset
# dataset.export(
#     export_dir=export_dir,
#     dataset_type=fo.types.ImageSegmentationDirectory,
#     label_field=label_field,
# )

session.wait()
