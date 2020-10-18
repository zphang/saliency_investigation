import tqdm
import pandas as pd

import torch
import torchvision.transforms as transforms

import casme.tasks.imagenet.utils as imagenet_utils
from casme.utils.torch_utils import ImagePathDataset
from casme.model_basics import casme_load_model

import zconf
from casme.tasks.imagenet.sanity_checks import get_scores, add_scores
import pyutils.io as io


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    casm_path1 = zconf.attr(default="best")
    casm_path2 = zconf.attr(default="best")
    val_json = zconf.attr(type=str)
    output_path = zconf.attr(default=None)


def main(args):
    data_loader = torch.utils.data.DataLoader(
        ImagePathDataset.from_path(
            config_path=args.val_json,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_utils.NORMALIZATION,
            ]),
            return_paths=True,
        ),
        batch_size=72, shuffle=False, num_workers=4, pin_memory=False
    )
    model1 = casme_load_model(
        args.casm_path1,
        classifier_load_mode="pickled",
        verbose=False,
    )
    model2 = casme_load_model(
        args.casm_path2,
        classifier_load_mode="pickled",
        verbose=False,
    )
    all_results = {}
    for i, ((input_, target), paths) in enumerate(tqdm.tqdm(data_loader)):
        input_ = input_.cuda()
        with torch.no_grad():
            classifier_output1, layers1 = model1['classifier'](input_, return_intermediate=True)
            masker_output1 = model1['masker'](layers1, use_p=None, class_ids=None, no_sigmoid=True)

            classifier_output2, layers2 = model2['classifier'](input_, return_intermediate=True)
            masker_output2 = model2['masker'](layers2, use_p=None, class_ids=None, no_sigmoid=True)
        score = get_scores(
            masker_output1.cpu().squeeze(1).numpy(),
            masker_output2.cpu().squeeze(1).numpy(),
            reduce=False,
        )
        add_scores(all_results, score)
    io.write_json(pd.DataFrame(all_results).mean().to_dict(), args.output_path)


if __name__ == "__main__":
    main(RunConfiguration.run_cli_json_prepend())
