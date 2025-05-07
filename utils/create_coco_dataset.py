import argparse
import os.path
from bisect import bisect_left
from typing import List

import fiftyone as fo
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from transformers.image_transforms import center_to_corners_format

from training_utils import COCO_ID2LABEL


def get_phrases_from_posmap(posmaps, input_ids, left_idx, right_idx):
    # Avoiding altering the input tensor
    posmaps = posmaps.clone()

    token_ids = []
    for posmap, l_idx, r_idx in zip(posmaps, left_idx, right_idx):
        posmap[0 : l_idx + 1] = False
        posmap[r_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids.append([input_ids[i] for i in non_zero_idx])

    return token_ids


def post_process_grounded_object_detection(
    processor,
    outputs,
    input_ids,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    target_sizes = None,
):
    logits, boxes = outputs.logits, outputs.pred_boxes
    sep_idx = [i for i in range(len(input_ids[0])) if input_ids[0][i].item() in [101, 102, 1012]]

    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    probs = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
    scores = torch.max(probs, dim=-1)[0]  # (batch_size, num_queries)

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(boxes)

    # Convert from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        if isinstance(target_sizes, List):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)  # noqa
        boxes = boxes * scale_fct[:, None, :]

    results = []
    for idx, (s, b, p) in enumerate(zip(scores, boxes, probs)):
        score = s[s > box_threshold]
        box = b[s > box_threshold]
        prob = p[s > box_threshold]

        max_idx = prob.argmax(dim=1)
        insert_idx = [bisect_left(sep_idx, idx) for idx in max_idx]
        right_idx = [sep_idx[idx] for idx in insert_idx]
        left_idx = [sep_idx[idx - 1] for idx in insert_idx]
        label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx], left_idx, right_idx)
        label = processor.batch_decode(label_ids)
        results.append({"scores": score, "labels": label, "boxes": box})

    return results


@torch.no_grad()
def main(
    image_directory: str,
    category_list: list[str],
    box_threshold: float,
    text_threshold: float,
    export_dataset: bool,
    export_path: str,
    view_dataset: bool,
    export_annotated_images: bool,
    model_id: str,
):
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id, local_files_only=True).to('cuda')
    dataset = fo.Dataset.from_images_dir(image_directory)

    text_prompt = ". ".join(category_list) + "."
    print("Input text prompt:", text_prompt)

    for sample in tqdm(dataset):
        image = Image.open(sample.filepath).convert('RGB')
        inputs = processor(images=image, text=text_prompt, return_tensors='pt').to('cuda')

        outputs = model(**inputs)
        results = post_process_grounded_object_detection(
            processor,
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []
        for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
            rel_box = torchvision.ops.box_convert(box, 'xyxy', 'xywh')  # noqa
            rel_box[0::2] /= image.size[0]
            rel_box[1::2] /= image.size[1]

            assert label in category_list, label
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=rel_box,
                    confidence=score,
                ))

        # Store detections in a field name of your choice
        sample["detections"] = fo.Detections(detections=detections)
        sample.save()

    # exports COCO dataset ready for training
    if export_dataset:
        dataset.export(
            export_path,
            dataset_type=fo.types.COCODetectionDataset,  # noqa
            classes=list(COCO_ID2LABEL.values())[1:]
        )

    # loads the voxel fiftyone UI ready for viewing the dataset.
    if view_dataset:
        session = fo.launch_app(dataset)
        session.wait()

    # saves bounding boxes plotted on the input images to disk
    if export_annotated_images:
        dataset.draw_labels(
            'images_with_bounding_boxes',
            label_fields=['detections']
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_directory",
        type=str,
        default='data/COCO-YOLO/images/content-counting_loss-th0.1-step1000-scale0.5',
        help="Directory containing images to be processed",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.4,
        help="Bounding box threshold",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.10,
        help="Text threshold",
    )
    parser.add_argument(
        "--export_dataset",
        type=bool,
        default=True,
        help="Export dataset to COCO format",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="data/Generation",
        help="Export path for COCO dataset",
    )
    parser.add_argument(
        "--view_dataset",
        type=bool,
        default=False,
        help="View dataset in FiftyOne",
    )
    parser.add_argument(
        "--export_annotated_images",
        type=bool,
        default=False,
        help="Export images with bounding boxes",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Hugging Face model ID",
    )
    args = parser.parse_args()
    args.export_path = os.path.join(args.export_path, args.image_directory.split('/')[-1])

    args.category_list = list(COCO_ID2LABEL.values())
    args.category_list = [category for category in args.category_list if category != 'N/A']

    main(
        args.image_directory,
        args.category_list,
        args.box_threshold,
        args.text_threshold,
        args.export_dataset,
        args.export_path,
        args.view_dataset,
        args.export_annotated_images,
        args.model_id
    )