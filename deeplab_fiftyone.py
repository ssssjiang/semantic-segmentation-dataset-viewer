import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "quickstart",
    dataset_name="segmentation-eval-demo",
    max_samples=10,
    shuffle=True,
)

CLASSES = (
        "background,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow," +
        "diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train," +
        "tvmonitor"
)
dataset.default_mask_targets = {
    idx: label for idx, label in enumerate(CLASSES.split(","))
}

model = foz.load_zoo_model("deeplabv3-resnet50-coco-torch")
dataset.apply_model(model, "semantic_segmentations")

session = fo.launch_app(dataset)

dataset.annotate(
    "segmentations",
    label_field="semantic_segmentations",
    launch_editor=True,
)

print(dataset.get_annotation_info("segmentations"))

dataset.load_annotations("segmentations")
session.refresh()

results = dataset.load_annotation_results("segmentations")
dataset.load_annotations("segmentations")

session.wait()