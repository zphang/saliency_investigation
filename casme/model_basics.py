import numpy as np
import scipy.ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion

import torch

from casme import archs
from casme.criterion import MaskFunc, default_infill_func
from dataclasses import dataclass
from typing import Union

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PARAMETERS = {
    "number_of_classes": 3,
    "num_filters": 16,
    "input_channels": 1,
    "first_layer_kernel_size": 7,
    "first_layer_conv_stride": 2,
    "first_pool_size": 3,
    "first_pool_stride": 2,
    "blocks_per_layer_list": [3, 4, 6, 3],
    "block_strides_list": [1, 2, 2, 2],
    "block_fn": "bottleneck",
}


@dataclass
class BoxCoords:
    # Exclusive

    xmin: Union[int, np.ndarray]
    xmax: Union[int, np.ndarray]
    ymin: Union[int, np.ndarray]
    ymax: Union[int, np.ndarray]

    @property
    def xslice(self):
        return slice(self.xmin, self.xmax)

    @property
    def yslice(self):
        return slice(self.ymin, self.ymax)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    def clamp(self, vmin, vmax):
        return self.__class__(
            xmin=np.clip(self.xmin, vmin, vmax),
            xmax=np.clip(self.xmax, vmin, vmax),
            ymin=np.clip(self.ymin, vmin, vmax),
            ymax=np.clip(self.ymax, vmin, vmax),
        )

    @classmethod
    def from_dict(cls, d):
        return cls(
            xmin=d["xmin"],
            xmax=d["xmax"],
            ymin=d["ymin"],
            ymax=d["ymax"],
        )

    def to_dict(self):
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }


def casme_load_model(casm_path, classifier_load_mode="pickled", verbose=True):
    name = casm_path.split('/')[-1].replace('.chk', '')

    if verbose:
        print("\n=> Loading model from '{}'".format(casm_path))
    checkpoint = torch.load(casm_path)

    if classifier_load_mode == "pickled":
        classifier = archs.resnet50shared()
        classifier.load_state_dict(checkpoint['state_dict_classifier'])
    elif classifier_load_mode == "original":
        classifier = archs.resnet50shared(pretrained=True)
    elif classifier_load_mode.startswith("path:"):
        classifier = archs.resnet50shared(pretrained=True, path=classifier_load_mode[5:])
    else:
        raise KeyError(classifier_load_mode)

    classifier.eval().to(device)

    if not isinstance(checkpoint["args"], dict):
        checkpoint["args"] = checkpoint["args"].__dict__
    masker = archs.default_masker(
        final_upsample_mode=checkpoint["args"].get("final_upsample_mode", "nearest"),
        add_prob_layers=checkpoint["args"].get("add_prob_layers", None),
        add_class_ids=checkpoint["args"].get("add_class_ids", None),
        apply_gumbel=checkpoint["args"].get("apply_gumbel", None),
        apply_gumbel_tau=checkpoint["args"].get("apply_gumbel_tau", None),
        # TOOD: fix argument resolution
        use_layers=archs.string_to_tuple(
            checkpoint["args"].get("masker_use_layers", "0,1,2,3,4"),
            cast=int,
        ),
    )
    if verbose:
        print(checkpoint["args"])
    if 'state_dict_masker' in checkpoint:
        masker.load_state_dict(checkpoint['state_dict_masker'])
    elif 'state_dict_decoder' in checkpoint:
        masker.load_state_dict(checkpoint['state_dict_decoder'])
        if verbose:
            print("Using old format")
    else:
        raise KeyError()
    masker.eval().to(device)
    if verbose:
        print("=> Model loaded.")

    return {'classifier': classifier, 'masker': masker, 'name': name, 'checkpoint': checkpoint}


def get_masks_and_check_predictions(input_, target, model, erode_k=0, dilate_k=0, use_p=None, no_sigmoid=False):
    with torch.no_grad():
        input_, target = input_.clone(), target.clone()
        mask, output = get_mask(
            input_=input_,
            model=model,
            use_p=use_p,
            get_output=True,
            no_sigmoid=no_sigmoid,
        )

        binarized_mask = binarize_mask(mask.clone())
        rectangular = torch.empty_like(binarized_mask)
        box_coord_ls = [BoxCoords(0, 0, 0, 0)] * len(input_)

        for idx in range(mask.size(0)):
            if binarized_mask[idx].sum() == 0:
                continue

            m = binarized_mask[idx].squeeze().cpu().numpy()
            if erode_k != 0:
                m = binary_erosion(m, iterations=erode_k, border_value=1)
            if dilate_k != 0:
                m = binary_dilation(m, iterations=dilate_k)
            rectangular[idx], box_coord_ls[idx] = get_rectangular_mask(m)

        target = target.to(device)
        _, max_indexes = output.data.max(1)
        is_correct = target.eq(max_indexes).long()

        return (
            mask.squeeze().cpu().numpy(),
            binarized_mask.cpu().numpy(),
            rectangular.squeeze().cpu().numpy(),
            is_correct.cpu().numpy(),
            box_coord_ls,
            output,
        )


def get_mask(input_, model, use_p=None, class_ids=None, get_output=False, no_sigmoid=False):
    with torch.no_grad():
        input_ = input_.to(device)
        classifier_output, layers = model['classifier'](input_, return_intermediate=True)
        masker_output = model['masker'](layers, use_p=use_p, class_ids=class_ids, no_sigmoid=no_sigmoid)
        if get_output:
            return masker_output, classifier_output
        else:
            return masker_output


def get_infilled(x, mask, infiller):
    with torch.no_grad():
        masked_x = MaskFunc.mask_out(x=x, mask=mask)
        generated = infiller(masked_x.detach(), mask.detach())
        infilled = default_infill_func(masked_x, mask, generated)
    return generated, infilled


def binarize_mask(mask):
    with torch.no_grad():
        batch_size = mask.size(0)
        avg = mask.view(batch_size, -1).mean(dim=1)
        binarized_mask = mask.gt(avg.view(batch_size, 1, 1, 1)).float()
        return binarized_mask.to(device)


def get_largest_connected(m):
    mask, num_labels = scipy.ndimage.label(m)
    largest_label = np.argmax(np.bincount(
        mask.reshape(-1), weights=m.reshape(-1)))
    largest_connected = (mask == largest_label)

    return largest_connected


def get_bounding_box(m):
    x = m.any(0)
    y = m.any(1)
    box_coords = BoxCoords(
        xmin=np.argmax(x),
        xmax=np.argmax(np.cumsum(x)),
        ymin=np.argmax(y),
        ymax=np.argmax(np.cumsum(y)),
    )
    with torch.no_grad():
        box_mask = torch.zeros(224, 224).to(device)
        box_mask[box_coords.yslice, box_coords.xslice] = 1

    return box_mask, box_coords


def get_rectangular_mask(m):
    return get_bounding_box(get_largest_connected(m))


def get_pred_bounding_box(rect):
    raw_y = np.arange(224)[rect.any(axis=0).astype(bool)]
    raw_x = np.arange(224)[rect.any(axis=1).astype(bool)]
    if len(raw_x) == 0 or len(raw_y) == 0:
        xmin, xmax = 0, 223
        ymin, ymax = 0, 223
    else:
        xmin, xmax = raw_x[0], raw_x[-1]
        ymin, ymax = raw_y[0], raw_y[-1]
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def get_saliency_point(single_soft_mask_arr, single_binary_mask_arr):
    # Need to check for mask in/out. Currently assuming mask in?
    mask_arr = single_soft_mask_arr * get_largest_connected(single_binary_mask_arr)
    point = tuple([int(np.round(i)) for i in scipy.ndimage.measurements.center_of_mass(mask_arr)])
    return point


def classification_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
