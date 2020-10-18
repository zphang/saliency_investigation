import copy
import math
import os
import tqdm

import numpy as np
import scipy.stats
import pandas as pd
import skimage.feature
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from casme.utils.torch_utils import ImagePathDataset
import casme.tasks.imagenet.utils as imagenet_utils
from casme.model_basics import casme_load_model, get_masks_and_check_predictions, device

import zconf
import pyutils.io as io


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    val_json = zconf.attr(help='train_json path')
    casm_path = zconf.attr(help='model_checkpoint')
    output_path = zconf.attr(help='output_path')

    layer_depth = zconf.attr(default=2, type=int)
    plot_img_i = zconf.attr(default=0, type=int)
    num_batches = zconf.attr(default=1, type=int)
    do_plot = zconf.attr(default=1, type=int)

    workers = zconf.attr(default=4, type=int, help='number of data loading workers (default: 4)')
    batch_size = zconf.attr(default=128, type=int, help='mini-batch size (default: 256)')
    break_ratio = zconf.attr(action='store_true', help='break original aspect ratio when resizing')


def main(args: RunConfiguration):
    data_loader = torch.utils.data.DataLoader(
        ImagePathDataset.from_path(
            config_path=args.val_json,
            transform=transforms.Compose([
                transforms.Resize([224, 224] if args.break_ratio else 224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_utils.NORMALIZATION,
            ]),
            return_paths=True,
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False
    )
    model = casme_load_model(args.casm_path)
    set_no_grad(model["classifier"])
    os.makedirs(args.output_path, exist_ok=True)

    all_results_dict = {
        "cascading": {},
        "independent": {},
    }
    plot_path_dict = {"cascading": [], "independent": []}
    for batch_i, ((input_, target), paths) in enumerate(tqdm.tqdm(
            data_loader, desc="batch", total=args.num_batches)):
        if batch_i >= args.num_batches:
            break
        img = get_image_arr(input_)[args.plot_img_i]
        plot_base_path = os.path.join(args.output_path, "plot__{}__{}".format(batch_i, args.plot_img_i))
        print(plot_base_path)

        input_ = input_.to(device)
        continuous, _, _, _, _, _ = get_masks_and_check_predictions(input_, target, model, no_sigmoid=True)
        gold_continuous = continuous
        scores = get_scores(gold_continuous, continuous, reduce=False)
        if args.do_plot:
            os.makedirs(plot_base_path, exist_ok=True)
            plot_file_name = "{}.png".format("normal")
            save_fig(
                img=img,
                mask=continuous[args.plot_img_i],
                title="normal",
                path=os.path.join(plot_base_path, plot_file_name),
            )
            plot_path_dict["cascading"].append(plot_file_name)
            plot_path_dict["independent"].append(plot_file_name)

        # Record normal scores
        if "normal" not in all_results_dict["cascading"]:
            all_results_dict["cascading"]["normal"] = {}
            all_results_dict["independent"]["normal"] = {}
        add_scores(d=all_results_dict["cascading"]["normal"], new_d=scores)
        add_scores(d=all_results_dict["independent"]["normal"], new_d=scores)

        # Cascading
        for layer_i, layer_name in enumerate(tqdm.tqdm(
                cascading_parameter_randomization_generator(model["classifier"], depth=args.layer_depth),
                desc="cascading")):
            if layer_name not in all_results_dict["cascading"]:
                all_results_dict["cascading"][layer_name] = {}
            continuous, _, _, _, _, _ = get_masks_and_check_predictions(input_, target, model, no_sigmoid=True)
            add_scores(
                d=all_results_dict["cascading"][layer_name],
                new_d=get_scores(gold_continuous, continuous, reduce=False)
            )
            if args.do_plot:
                plot_file_name = "cascading__{:02d}__.png".format(layer_i, layer_name)
                save_fig(
                    img=img,
                    mask=continuous[args.plot_img_i],
                    title=layer_name,
                    path=os.path.join(plot_base_path, plot_file_name),
                )
                plot_path_dict["cascading"].append(plot_file_name)

        # Independent
        for layer_i, layer_name in enumerate(tqdm.tqdm(
                independent_parameter_randomization_generator(model["classifier"], depth=args.layer_depth),
                desc="independent")):
            if layer_name not in all_results_dict["independent"]:
                all_results_dict["independent"][layer_name] = {}
            continuous, binarized_mask, rectangular, is_correct, bboxes, classifier_outputs = \
                get_masks_and_check_predictions(input_, target, model)
            add_scores(
                d=all_results_dict["independent"][layer_name],
                new_d=get_scores(gold_continuous, continuous, reduce=False)
            )
            if args.do_plot:
                plot_file_name = "independent__{:02d}__.png".format(layer_i, layer_name)
                save_fig(
                    img=img,
                    mask=continuous[args.plot_img_i],
                    title=layer_name,
                    path=os.path.join(plot_base_path, plot_file_name),
                )
                plot_path_dict["independent"].append(plot_file_name)
        write_html(plot_path_dict, plot_base_path)

    results = compile_results(all_results_dict)
    io.write_json(results, os.path.join(args.output_path, "scores.json"))


def write_html(plot_path_dict, plot_base_path):
    for mode in ["cascading", "independent"]:
        html = "<html><body>\n"
        for path in plot_path_dict[mode]:
            html += "<img src='{}' style='width:100%'>".format(path)
        html += "</html></body>\n"
        io.write_file(html, os.path.join(plot_base_path, "{}.html".format(mode)))


def get_image_arr(input_):
    return np.clip(imagenet_utils.denormalize_arr(imagenet_utils.tensor_to_image_arr(input_)), 0, 1)


def compile_results(all_results_dict):
    return{
        sanity_type: {
            layer_name: {
                k: float(np.mean(v))
                for k, v in scores_dict.items()
            }
            for layer_name, scores_dict in sanity_type_results.items()
        }
        for sanity_type, sanity_type_results in all_results_dict.items()
    }


def add_scores(d, new_d):
    for k, v in new_d.items():
        if k not in d:
            d[k] = []
        d[k] += v


def copy_state_dict(model):
    return copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})


def set_no_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def get_parametered_layer_names(model, depth=1):
    candidate_list = []
    for name, module in model.named_modules():
        if not name:
            continue
        if len(name.split(".")) > depth:
            continue
        if not len(list(module.parameters())):
            continue
        for other_name in list(candidate_list):
            if name.startswith(other_name):
                candidate_list.remove(other_name)
                break
        candidate_list.append(name)
    return candidate_list


def chain_getattr(obj, attr_name):
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def randomize_layer(layer):
    # Just taking random initialization schemes
    for name, param in layer.named_parameters():
        if len(param.shape) == 4:
            # Conv kernel
            n = param.shape[0] * param.shape[1] * param.shape[3]
            param.normal_(0, math.sqrt(2. / n))
        elif len(param.shape) == 2:
            # Linear weight
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif len(param.shape) == 1:
            # Linear bias
            bound = 1 / math.sqrt(len(param))
            torch.nn.init.uniform_(param, -bound, bound)
        else:
            raise RuntimeError()


def get_model_device(model):
    parameter_list = list(model.parameters())
    if parameter_list:
        return parameter_list[0].device
    else:
        return torch.device("cpu")


def cascading_parameter_randomization_generator(model, depth=1):
    device = get_model_device(model)
    state_dict_copy = copy_state_dict(model)
    reversed_layer_names = get_parametered_layer_names(model, depth=depth)[::-1]
    for layer_name in reversed_layer_names:
        layer = chain_getattr(model, layer_name)
        randomize_layer(layer)
        yield layer_name
    model.load_state_dict(state_dict_copy)
    model.to(device)


def independent_parameter_randomization_generator(model, depth=1):
    device = get_model_device(model)
    reversed_layer_names = get_parametered_layer_names(model, depth=depth)[::-1]
    for layer_name in reversed_layer_names:
        layer = chain_getattr(model, layer_name)
        layer_state_dict_copy = copy_state_dict(layer)
        randomize_layer(layer)
        yield layer_name
        layer.load_state_dict(layer_state_dict_copy)
        layer.to(device)


def spearman_comparison(masks1, masks2, do_abs=False, reduce=True):
    assert masks1.shape == masks2.shape
    ls = []
    for i in range(masks1.shape[0]):
        mask1 = masks1[i]
        mask2 = masks2[i]
        if do_abs:
            mask1 = np.abs(0.5 - mask1)
            mask2 = np.abs(0.5 - mask2)

        correl = scipy.stats.spearmanr(
            mask1.reshape(-1),
            mask2.reshape(-1),
        ).correlation
        ls.append(correl)
    if reduce:
        return pd.Series(ls).mean()  # ignores nans, which arise from all=1 masks
    else:
        return ls


def get_hog_correl(masks1, masks2, reduce=True):
    assert masks1.shape == masks2.shape
    ls = []
    for i in range(masks1.shape[0]):
        mask1 = masks1[i]
        mask2 = masks2[i]

        correl = scipy.stats.pearsonr(
            skimage.feature.hog(mask1, pixels_per_cell=(16, 16)),
            skimage.feature.hog(mask2, pixels_per_cell=(16, 16)),
        )[0]
        ls.append(correl)
    if reduce:
        return pd.Series(ls).mean()  # ignores nans, which arise from all=1 masks
    else:
        return ls


def get_ssim(masks1, masks2, reduce=True):
    assert masks1.shape == masks2.shape
    ls = []
    for i in range(masks1.shape[0]):
        mask1 = masks1[i]
        mask2 = masks2[i]
        ls.append(ssim(mask1, mask2, win_size=5))
    if reduce:
        return np.mean(ls)
    else:
        return ls


def get_scores(masks1, masks2, reduce=True):
    return {
        "spearman": spearman_comparison(masks1, masks2, do_abs=False, reduce=reduce),
        "abs_spearman": spearman_comparison(masks1, masks2, do_abs=True, reduce=reduce),
        "hog_correl": get_hog_correl(masks1, masks2, reduce=reduce),
        "ssim": get_ssim(masks1, masks2, reduce=reduce),
    }


def save_fig(img, mask, title, path):
    # img = image_arrs[img_i]
    # mask = continuous[img_i]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img)
    axes[1].imshow(mask)
    axes[2].imshow((img * mask[:, :, np.newaxis]))
    axes[3].imshow((img * (1 - mask[:, :, np.newaxis])))
    plt.suptitle(title, fontsize=16)
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
