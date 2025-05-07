import os
import pickle
import sys

from PIL import Image
from accelerate import PartialState
from tqdm import tqdm
from transformers import pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from grounded_sam.dataclass import DetectionResult
from utils.training_utils import COCO_ID2LABEL

distributed_state = PartialState()
img_path = "data/COCO/val2017"
save_path = "data/COCO/annotations_gd/val2017"
labels = list(filter(lambda x: x != "N/A", COCO_ID2LABEL.values()))
assert len(labels) == 80
threshold = 0.2
detector_id = "IDEA-Research/grounding-dino-base"
object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=distributed_state.device)
image_list = sorted(os.listdir(img_path))

with distributed_state.split_between_processes(image_list) as split_list:
    for i_path in tqdm(split_list):
        path = os.path.join(img_path, i_path)
        img = Image.open(path)

        # detect objects
        results = object_detector(img, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        # save results
        i_save_path = os.path.join(save_path, i_path.replace(".jpg", ".pkl"))
        i_file = open(i_save_path, "wb")
        pickle.dump(results, i_file)
        i_file.close()
