import json
from collections import defaultdict
from pathlib import Path

import datasets

_CLASSES = [
    "None", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"
]


class COCOHelper:
    def __init__(self, captions_path, instances_path):
        self.caps, self.insts, self.imgs = dict(), dict(), dict()
        self.path2img, self.img2captions, self.img2instances = defaultdict(list), defaultdict(list), defaultdict(list)

        with open(captions_path, "r") as cp:
            captions = json.load(cp)
        with open(instances_path, "r") as ip:
            instances = json.load(ip)
        self.captions = captions
        self.instances = instances

        self.create_index()

    def create_index(self):
        # double check image id
        for image in self.captions["images"]:
            self.imgs[image["id"]] = image
        for image in self.instances["images"]:
            assert self.imgs[image["id"]] == image

        # create img path index
        for img in self.captions["images"]:
            self.path2img[img["file_name"]] = img["id"]

        # create captions index
        for cap in self.captions["annotations"]:
            self.img2captions[cap["image_id"]].append(cap)
            self.caps[cap["id"]] = cap

        # create instances index
        for inst in self.instances["annotations"]:
            self.img2instances[inst["image_id"]].append(inst)
            self.insts[inst["id"]] = inst

    def __len__(self):
        return len(self.captions["images"])

    def get_img_id(self, img_name):
        return self.path2img.get(img_name)

    def get_captions(self, img_id):
        return self.img2captions.get(img_id)

    def get_instances(self, img_id):
        return self.img2instances.get(img_id, [])


class COCO2017(datasets.GeneratorBasedBuilder):
    """COCO 2017 dataset with captions and annotations."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),
                "image_id": datasets.Value("int64"),
                "file_name": datasets.Value("string"),
                "height": datasets.Value("int64"),
                "width": datasets.Value("int64"),
                "captions": [{
                    "id": datasets.Value("int64"),
                    "caption": datasets.Value("string"),
                }],
                "instances": [{
                    "id": datasets.Value("int64"),
                    "area": datasets.Value("float64"),
                    "bbox": datasets.Sequence(datasets.Value("float32"), length=4),
                    "label": datasets.ClassLabel(names=_CLASSES),
                    "iscrowd": datasets.Value("bool"),
                }],
            }),
        )

    def _split_generators(self, dl_manager):
        data_root = Path('data/COCO')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "captions_path": data_root / 'annotations/captions_train2017.json',
                    "instances_path": data_root / 'annotations/instances_train2017.json',
                    "images": dl_manager.iter_files(str(data_root / 'train2017')),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "captions_path": data_root / 'annotations/captions_val2017.json',
                    "instances_path": data_root / 'annotations/instances_val2017.json',
                    "images": dl_manager.iter_files(str(data_root / 'val2017')),
                },
            )
        ]

    def _generate_examples(self, captions_path, instances_path, images):
        counter = 0
        helper = COCOHelper(captions_path, instances_path)

        for img_path in images:
            file_name = Path(img_path).name
            img_id = helper.get_img_id(file_name)
            yield counter, {
                "image": str(Path(img_path).absolute()),
                "image_id": img_id,
                "file_name": file_name,
                "height": helper.imgs[img_id]["height"],
                "width": helper.imgs[img_id]["width"],
                "captions": [
                    {
                        "id": cap["id"],
                        "caption": cap["caption"],
                    }
                    for cap in helper.get_captions(img_id)
                ],
                "instances": [
                    {
                        "id": inst["id"],
                        "area": inst["area"],
                        "bbox": inst["bbox"],  # [x, y, w, h]
                        "label": inst["category_id"],
                        "iscrowd": bool(inst["iscrowd"]),
                    }
                    for inst in helper.get_instances(img_id)
                ],
            }
            counter += 1
