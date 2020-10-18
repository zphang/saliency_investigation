import os
import numpy as np
import zconf
import pyutils.io as io
from scipy.stats import rankdata
from scipy.stats import multivariate_normal
from tqdm import auto as tqdm_lib

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import casme.model_basics
import casme.tasks.imagenet.utils as imagenet_utils
import casme.utils.results_utils as results_utils
from casme import archs
from casme.utils.torch_utils import ImagePathDataset

from evaluation import configure_metadata, MaskEvaluator, BoxEvaluator, get_image_ids, t2n, ospj


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    cam_loader = zconf.attr(type=str, required=True)
    casm_base_path = zconf.attr(type=str, default=None)
    output_base_path = zconf.attr(type=str, default=None)

    dataset = zconf.attr(type=str)
    dataset_split = zconf.attr(type=str)
    dataset_path = zconf.attr(type=str)
    metadata_path = zconf.attr(type=str)
    cam_curve_interval = zconf.attr(type=float, default=0.01)
    box_v2_metric = zconf.attr(action="store_true")

    classifier_load_mode = zconf.attr(default="pickled")
    workers = zconf.attr(default=4, type=int, help='number of data loading workers (default: 4)')
    batch_size = zconf.attr(default=128, type=int, help='mini-batch size (default: 256)')
    break_ratio = True

    # === Dataset-specific === #
    # Used for ILSVRC/test
    imagenet_val_path = zconf.attr(type=str, default=None)

    # === Method-specific === #
    torchray_method = zconf.attr(default=None)
    casme_load_mode = zconf.attr(type=str, default="best")


def zeros_cam_loader_getter(image_ids):
    for image_id in image_ids:
        yield torch.zeros([224, 224]), image_id


def ones_cam_loader_getter(image_ids):
    for image_id in image_ids:
        yield torch.ones([224, 224]), image_id


def center_cam_loader_getter(image_ids):
    for image_id in image_ids:
        x = torch.zeros([224, 224])
        x[33:191, 33:191] = 1
        yield x, image_id


def gaussian_cam_loader_getter(image_ids):
    x = np.arange(-112, 112, 1)
    y = np.arange(-112, 112, 1)
    xy = np.stack(np.meshgrid(x, y)[::-1], axis=2)
    var = multivariate_normal(mean=[0, 0], cov=[[112 ** 2, 0], [0, 112 ** 2]])
    mask = torch.tensor(rank_and_normalize(var.pdf(xy)))
    for image_id in image_ids:
        yield mask, image_id


def rank_and_normalize(x):
    ranked = rankdata(x).reshape(*x.shape)
    ranked = (ranked - ranked.min()) / (ranked.max() - ranked.min())
    return ranked


class GenerationCamLoader:
    def __init__(self, args: RunConfiguration):
        self.args = args
        self.workers = args.workers
        self.batch_size = args.batch_size
        self.break_ratio = args.break_ratio
        self.dataset = args.dataset
        self.dataset_split = args.dataset_split

    def getter(self, image_ids):
        if self.dataset == "ILSVRC" and self.dataset_split == "test":
            dataset_config = io.read_json(self.args.imagenet_val_path)
            path_to_image_id_dict = {
                x[0]: "val/" + x[0].split("/")[-1]
                for x in dataset_config["samples"]
            }
        else:
            dataset_root = os.path.join(self.args.dataset_path, self.dataset)
            image_paths_and_labels = []
            for image_id in image_ids:
                dummy_label = 0
                image_paths_and_labels.append((
                    os.path.join(dataset_root, image_id),
                    dummy_label,
                ))
            dataset_config = {
                "root": dataset_root,
                "samples": image_paths_and_labels,
                "classes": None,
                "class_to_idx": None,
            }
            path_to_image_id_dict = None

        data_loader = torch.utils.data.DataLoader(
            ImagePathDataset(
                config=dataset_config,
                transform=transforms.Compose([
                    transforms.Resize([224, 224] if self.break_ratio else 224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    imagenet_utils.NORMALIZATION,
                ]),
                return_paths=True,
            ),
            batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=False
        )
        for i, ((input_, target), paths) in enumerate(data_loader):
            mask = self.get_mask(input_=input_, target=target)
            mask = mask.detach().cpu().squeeze(1)
            for j, single_mask in enumerate(mask):
                if self.dataset == "ILSVRC" and self.dataset_split == "test":
                    image_id = path_to_image_id_dict[paths[j]]
                else:
                    image_id = paths[j][len(data_loader.dataset.root) + 1:]
                yield single_mask, image_id

    def get_mask(self, input_, target):
        raise NotImplementedError()


class CasmeCamLoader(GenerationCamLoader):
    def __init__(self, args: RunConfiguration):
        super().__init__(args=args)
        if args.casme_load_mode == "best":
            casm_path = results_utils.find_best_model(args.casm_base_path)
        elif args.casme_load_mode == "specific":
            casm_path = args.casm_base_path
        else:
            raise KeyError(args.casme_load_mode)
        self.model = casme.model_basics.casme_load_model(
            casm_path=casm_path,
            classifier_load_mode=args.classifier_load_mode,
        )

    def get_mask(self, input_, target):
        mask, output = casme.model_basics.get_mask(
            input_=input_,
            model=self.model,
            use_p=None,
            get_output=True,
            no_sigmoid=False,
        )
        return mask


class TorchrayCamLoader(GenerationCamLoader):
    def __init__(self, args: RunConfiguration):
        super().__init__(args=args)
        self.torchray_method = args.torchray_method
        self.grad_cam_upsampler = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.original_classifier = archs.resnet50shared(pretrained=True)

    def get_mask(self, input_, target):
        saliency_ls = []
        for j in range(len(target)):
            input_single = input_[j:j + 1]
            target_single = target[j].item()
            if self.torchray_method == "grad_cam":
                from torchray.attribution.grad_cam import grad_cam
                saliency = grad_cam(
                    model=self.original_classifier,
                    input=input_single,
                    target=target_single,
                    saliency_layer='layer4',
                )
                saliency = self.grad_cam_upsampler(saliency)
            elif self.torchray_method == "guided_backprop":
                from torchray.attribution.guided_backprop import guided_backprop
                saliency = guided_backprop(
                    model=self.original_classifier,
                    input=input_single,
                    target=target_single,
                    resize=(224, 224),
                    smooth=0.02,
                )
            else:
                raise KeyError()
            if saliency.max() == saliency.min():
                saliency[:] = 1
            else:
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            saliency_ls.append(saliency.detach())
        return torch.cat(saliency_ls, dim=0)


def get_cam_loader_getter(args: RunConfiguration):
    if args.cam_loader == "zeros":
        return zeros_cam_loader_getter
    elif args.cam_loader == "ones":
        return ones_cam_loader_getter
    elif args.cam_loader == "center":
        return center_cam_loader_getter
    elif args.cam_loader == "gaussian":
        return gaussian_cam_loader_getter
    elif args.cam_loader == "casme":
        return CasmeCamLoader(args).getter
    elif args.cam_loader == "torchray":
        return TorchrayCamLoader(args).getter
    else:
        raise KeyError()


def evaluate_wsol_from_cam_loader(
        cam_loader_getter, metadata_root, mask_root, dataset_name, split,
        multi_contour_eval, multi_iou_eval, iou_threshold_list,
        cam_curve_interval=.001):
    print("Loading and evaluating cams.")
    meta_path = os.path.join(metadata_root, dataset_name, split)
    metadata = configure_metadata(meta_path)
    image_ids = get_image_ids(metadata)
    cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

    evaluator = {"OpenImages": MaskEvaluator,
                 "CUB": BoxEvaluator,
                 "ILSVRC": BoxEvaluator
                 }[dataset_name](metadata=metadata,
                                 dataset_name=dataset_name,
                                 split=split,
                                 cam_threshold_list=cam_threshold_list,
                                 mask_root=ospj(mask_root, 'OpenImages'),
                                 multi_contour_eval=multi_contour_eval,
                                 iou_threshold_list=iou_threshold_list)
    cam_loader = cam_loader_getter(image_ids)
    for cam, image_id in tqdm_lib.tqdm(cam_loader):
        evaluator.accumulate(t2n(cam), image_id)
    performance = evaluator.compute()
    if multi_iou_eval or dataset_name == 'OpenImages':
        performance = np.average(performance)
    else:
        performance = performance[iou_threshold_list.index(50)]

    print('localization: {}'.format(performance))
    return performance


def main(args: RunConfiguration):
    cam_loader_getter = get_cam_loader_getter(args=args)
    if args.box_v2_metric:
        multi_contour_eval = True
        multi_iou_eval = True
        iou_threshold_list = [30, 50, 70]
    else:
        multi_contour_eval = False
        multi_iou_eval = False
        iou_threshold_list = [50]

    performance = evaluate_wsol_from_cam_loader(
        cam_loader_getter=cam_loader_getter,
        metadata_root=args.metadata_path,
        mask_root=args.dataset_path,
        dataset_name=args.dataset,
        split=args.dataset_split,
        multi_contour_eval=multi_contour_eval,
        multi_iou_eval=multi_iou_eval,
        iou_threshold_list=iou_threshold_list,
        cam_curve_interval=args.cam_curve_interval,
    )
    if args.output_base_path:
        output_base_path = args.output_base_path
    else:
        output_base_path = os.path.join(args.casm_base_path, "wsoleval")
    os.makedirs(output_base_path, exist_ok=True)
    file_name = "{}___{}___{}.json".format(
        args.dataset,
        args.dataset_split,
        "v2" if args.box_v2_metric else "v1",
    )
    io.write_json({
        "performance": performance,
    }, os.path.join(output_base_path, file_name))


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
