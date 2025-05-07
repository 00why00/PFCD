import argparse
import os

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator


class LabelCountValidator(DetectionValidator):
    """
    LabelCountValidator.
    The validator for mLC metric.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iouv = torch.linspace(0.3, 0.95, 14, device=self.device)
        self.niou = self.iouv.numel()

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        return {"cls": cls}

    def _prepare_pred(self, pred, **kwargs):
        """Remove bbox from pred."""
        pred = pred[:, 4:]
        return pred

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls = pbatch.pop("cls")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred)
            stat["conf"] = predn[:, 0]
            stat["pred_cls"] = predn[:, 1]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def match_predictions(self, pred_classes, true_classes, confidence, **kwargs):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            confidence (torch.Tensor): Confidence class indices of shape(N,).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,19) for 19 IoU thresholds.
        """
        # Dx19 matrix, where D - classes, 19 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        confidence = confidence * correct_class  # zero out the wrong classes
        confidence = confidence.cpu().numpy()
        mark = 1e9
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            cost_mat = np.array(confidence >= threshold).astype(int)
            cost_mat[np.where(cost_mat == 0)] = mark
            index, target = linear_sum_assignment(cost_mat)
            # remove failed matches from target
            target = target[cost_mat[index, target] != mark]
            correct[target, i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _process_batch(self, pred, gt_cls, **kwargs):
        """
        Return correct prediction matrix.

        Args:
            pred (torch.Tensor): Tensor of shape [N, 2] representing classes.
                Each class is of the format: conf, class.
            gt_cls (torch.Tensor): Tensor of shape [M] representing labels.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 19] for 19 IoU levels.
        """
        return self.match_predictions(pred[:, 1], gt_cls, pred[:, 0])


class Evaluator:
    def __init__(self, args):
        self.eval_model = YOLO(os.path.join("./downstream_tasks", "yolov8m.pt" if args.dataset == "coco" else "yolov8m_voc.pt"))
        self.train_name = args.train_name

        # prepare image list for eval
        if args.dataset == "coco":
            standard_image_list_path = os.path.join(args.eval_output_path, "val2017.txt")
            image_list_path = os.path.join(args.eval_output_path, f"{args.train_name}.txt")
            standard_image_list = open(standard_image_list_path, "r").read()
            new_image_list = standard_image_list.replace("val2017", args.train_name)
            new_image_list_file = open(image_list_path, "w")
            new_image_list_file.write(new_image_list)
            new_image_list_file.flush()

            # prepare yaml file for eval
            coco_config_file_path = os.path.join("./downstream_tasks", "coco.yaml")
            coco_config_file = open(coco_config_file_path, "r")
            data_config = yaml.load(coco_config_file.read(), Loader=yaml.FullLoader)
            data_config["val"] = f"{args.train_name}.txt"
            self.new_config_file_path = os.path.join(args.eval_output_path, f"{args.train_name}.yaml")
            new_config_file = open(self.new_config_file_path, "w")
            yaml.dump(data_config, new_config_file)

            # prepare labels cache for eval
            self.labels_cache_path = os.path.join(args.eval_output_path, "labels")
            os.rename(os.path.join(self.labels_cache_path, "val2017"), os.path.join(self.labels_cache_path, args.train_name))
        else:
            voc_config_file_path = os.path.join("./downstream_tasks", "voc.yaml")
            voc_config_file = open(voc_config_file_path, "r")
            data_config = yaml.load(voc_config_file.read(), Loader=yaml.FullLoader)
            data_config["val"] = f"images/{args.train_name}"
            self.new_config_file_path = os.path.join(args.eval_output_path, f"{args.train_name}.yaml")
            new_config_file = open(self.new_config_file_path, "w")
            yaml.dump(data_config, new_config_file)

            self.labels_cache_path = os.path.join(args.eval_output_path, "labels")
            os.rename(os.path.join(self.labels_cache_path, "test2007"), os.path.join(self.labels_cache_path, args.train_name))

    def evaluate(self):
        metrics = self.eval_model.val(
            mode='val',
            data=self.new_config_file_path,
            validator=LabelCountValidator,
            plots=False,
            project='output_evaluation',
            name=self.train_name
        )
        print(metrics.box.map)
        print(metrics.box.all_ap.mean(axis=0))
        print(metrics.box.all_ap.mean(axis=0)[5])

    def __del__(self):
        os.rename(os.path.join(self.labels_cache_path, self.train_name), os.path.join(self.labels_cache_path, "val2017"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_name",
        type=str,
        default=None,
        help="The path to the lora weights."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        help="The dataset to evaluate."
    )
    parser.add_argument(
        "--eval_output_path",
        type=str,
        default="data/{}-YOLO",
        help="The path to the evaluation output."
    )
    args = parser.parse_args()
    args.eval_output_path = args.eval_output_path.format(args.dataset.upper())

    evaluator = Evaluator(args)
    evaluator.evaluate()


if __name__ == '__main__':
    main()