import cv2
import numpy as np

import torch
import torchvision.transforms as transforms

from casme.model_basics import binarize_mask, get_mask, get_saliency_point


def get_binarized_mask(input, model, use_p=None, class_ids=None):
    mask = get_mask(input, model, use_p=use_p, class_ids=class_ids)
    return binarize_mask(mask.clone()), mask


def get_masked_images(images, binary_mask, gray_scale=0, return_mask=False):
    with torch.no_grad():
        if gray_scale > 0:
            gray_background = torch.zeros_like(images) + 0.35
            masked_in = binary_mask * images + (1 - binary_mask) * gray_background
            masked_out = (1 - binary_mask) * images + binary_mask * gray_background
        else:
            masked_in = binary_mask * images
            masked_out = (1 - binary_mask) * images

        if return_mask:
            return masked_in, masked_out, binary_mask
        else:
            return masked_in, masked_out


def get_masked_images_v2(batch_x, mask):
    with torch.no_grad():
        masked_in = batch_x * mask
        masked_out = batch_x * (1 - mask)
    return masked_in, masked_out


def inpaint(mask, masked_image):
    l = []
    for i in range(mask.size(0)):
        permuted_image = permute_image(masked_image[i], mul255=True)
        m = mask[i].squeeze().byte().cpu().numpy()
        inpainted_numpy = cv2.inpaint(permuted_image, m, 3, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
        l.append(transforms.ToTensor()(inpainted_numpy).unsqueeze(0))
    inpainted_tensor = torch.cat(l, 0)

    return inpainted_tensor       


def permute_image(image_tensor, mul255=False):
    with torch.no_grad():
        image = image_tensor.clone().squeeze().permute(1, 2, 0)
        if mul255:
            image *= 255
            image = image.byte()

        return image.cpu().numpy()


def per_image_normalization(x, mode):
    if mode is None:
        return x
    assert len(x.shape) == 4
    x_max = x.view(x.shape[0], -1).max(1)[0].view(-1, 1, 1, 1) + 1e-6
    x_min = x.view(x.shape[0], -1).min(1)[0].view(-1, 1, 1, 1)
    if mode == "-1_1":
        normalized_x = (x - x_min) / (x_max - x_min).view(-1, 1, 1, 1) * 2 - 1
    elif mode == "0_1":
        normalized_x = (x - x_min) / (x_max - x_min).view(-1, 1, 1, 1)
    else:
        raise KeyError(mode)
    return normalized_x


def count_params(module):
    return sum(
        np.prod(param.shape)
        for param in module.parameters()
    )


def arr_for_plot(tensor):
    return tensor.detach().permute(1, 2, 0).cpu().numpy()


class CasmeModelWrapper:
    def __init__(self, model_dict):
        self.model_dict = model_dict

    def get_saliency_points(self, x, y):
        binary_mask, soft_mask = get_binarized_mask(x, self.model_dict, class_ids=y)
        binary_mask_arr = binary_mask.squeeze(1).cpu().numpy()
        soft_mask_arr = soft_mask.squeeze(1).cpu().numpy()
        saliency_point_ls = []
        for i in range(x.shape[0]):
            saliency_point_ls.append(get_saliency_point(
                single_soft_mask_arr=soft_mask_arr[i],
                single_binary_mask_arr=binary_mask_arr[i],
            ))
        return saliency_point_ls
