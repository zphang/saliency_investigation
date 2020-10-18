import time
import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from casme import core, archs, criterion
from casme.train_utils import single_adjust_learning_rate, save_checkpoint, set_args
from casme.tasks.imagenet.utils import get_data_loaders

import zconf
import zproto.zlogv1 as zlog


@zconf.run_config
class RunConfiguration(zconf.RunConfig):

    # Experiment Setup
    train_json = zconf.attr(help='train_json path')
    val_json = zconf.attr(help='val_json path')
    output_path = zconf.attr(help='output_path')
    name = zconf.attr(default='random',
                      help='name used to build a path where the models and log are saved (default: random)')
    log_buffer = zconf.attr(default=10, type=int, help='log buffer')
    workers = zconf.attr(default=4, type=int,
                         help='number of data loading workers (default: 4)')

    epochs = zconf.attr(default=60, type=int,
                        help='number of total epochs to run')
    batch_size = zconf.attr(default=128, type=int,
                            help='mini-batch size (default: 128)')
    perc_of_training = zconf.attr(default=0.2, type=float,
                                  help='percent of training set seen in each epoch')
    do_val = zconf.attr(action="store_true")
    lr = zconf.attr(default=0.001, type=float,
                    help='initial learning rate for classifier')
    lr_casme = zconf.attr(default=0.001, type=float,
                          help='initial learning rate for casme')
    lrde = zconf.attr(default=20, type=int,
                      help='how often is the learning rate decayed')
    momentum = zconf.attr(default=0.9, type=float,
                          help='momentum for classifier')
    weight_decay = zconf.attr(default=1e-4, type=float,
                              help='weight decay for both classifier and casme (default: 1e-4)')

    upsample = zconf.attr(default='nearest',
                          help='mode for final upsample layer in the decoder (default: nearest)')
    fixed_classifier = zconf.attr(action='store_true',
                                  help='train classifier')
    prob_historic = zconf.attr(default=0.5, type=float,
                               help='probability for evaluating historic model')
    save_freq = zconf.attr(default=1000, type=int,
                           help='frequency of model saving to history (in batches)')
    actual_save_freq = zconf.attr(default=1, type=int)
    f_size = zconf.attr(default=30, type=int,
                        help='size of F set - maximal number of previous classifier iterations stored')
    resnet_path = zconf.attr(default=None, type=str, help="If none, defaults to loading from full ResNet-50")
    lambda_r = zconf.attr(default=None, type=float)
    lambda_tv = zconf.attr(default=None, type=float)

    masker_use_layers = zconf.attr(default="0,1,2,3,4", type=str)

    mask_in_criterion = zconf.attr(default="none", type=str, help='crossentropy|kldivergence|none')
    mask_in_criterion_config = zconf.attr(default="", type=str, help='etc')
    mask_in_objective_direction = zconf.attr(default="maximize", help="maximize|minimize")
    mask_in_objective_type = zconf.attr(default="entropy", help="entropy|classification")
    mask_in_weight = zconf.attr(default=1.0, type=float)
    mask_in_lambda_r = zconf.attr(default=10, type=float)
    mask_in_lambda_tv = zconf.attr(default=None, type=float)

    mask_out_criterion = zconf.attr(default="none", type=str, help='crossentropy|kldivergence|none')
    mask_out_criterion_config = zconf.attr(default="", type=str, help='etc')
    mask_out_objective_direction = zconf.attr(default="maximize", help="maximize|minimize")
    mask_out_objective_type = zconf.attr(default="entropy", help="entropy|classification")
    mask_out_weight = zconf.attr(default=1.0, type=float)
    mask_out_lambda_r = zconf.attr(default=10, type=float)
    mask_out_lambda_tv = zconf.attr(default=None, type=float)

    reproduce = zconf.attr(default='',
                           help='reproducing paper results (F|L|FL|L100|L1000)')

    add_prob_layers = zconf.attr(action='store_true')
    prob_sample_low = zconf.attr(default=0.25, type=float)
    prob_sample_high = zconf.attr(default=0.75, type=float)
    prob_loss_func = zconf.attr(default="l1")
    add_class_ids = zconf.attr(action='store_true')
    apply_gumbel = zconf.attr(action='store_true')
    apply_gumbel_tau = zconf.attr(default=0.1, type=float)
    gumbel_output_mode = zconf.attr(default="hard", type=str)

    # Placeholders
    casms_path = zconf.attr(default='')
    log_path = zconf.attr(default='')

    infiller_model = zconf.attr(default=None, type=str)
    do_infill_for_mask_in = zconf.attr(default=0, type=int)
    do_infill_for_mask_out = zconf.attr(default=0, type=int)

    def _post_init(self):
        randomhash = ''.join(str(time.time()).split('.'))
        self.name = self.name + "___" + randomhash
        self.need_infiller = self.do_infill_for_mask_in or self.do_infill_for_mask_out
        if self.lambda_r is not None:
            assert self.mask_in_lambda_r == 10
            assert self.mask_out_lambda_r == 10
            self.mask_in_lambda_r = self.lambda_r
            self.mask_out_lambda_r = self.lambda_r
        if self.lambda_tv is not None:
            assert self.mask_in_lambda_tv is None
            assert self.mask_out_lambda_tv is None
            self.mask_in_lambda_tv = self.lambda_tv
            self.mask_out_lambda_tv = self.lambda_tv
        if self.resnet_path == "none":
            self.resnet_path = None

        set_args(self)


def main(args):
    print("Path: {}".format(args.casms_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create models and optimizers
    classifier = archs.resnet50shared(pretrained=True, path=args.resnet_path).to(device)
    masker = archs.default_masker(
        final_upsample_mode=args.upsample,
        add_prob_layers=args.add_prob_layers,
        add_class_ids=args.add_class_ids,
        apply_gumbel=args.apply_gumbel,
        apply_gumbel_tau=args.apply_gumbel_tau,
        gumbel_output_mode=args.gumbel_output_mode,
        use_layers=archs.string_to_tuple(args.masker_use_layers, cast=int),
    ).to(device)
    classifier_optimizer = torch.optim.SGD(
        classifier.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    masker_optimizer = torch.optim.Adam(
        masker.parameters(), args.lr_casme,
        weight_decay=args.weight_decay,
    )

    cudnn.benchmark = True

    train_loader, val_loader = get_data_loaders(
        train_json=args.train_json,
        val_json=args.val_json,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    mask_in_criterion = criterion.resolve_masker_criterion(
        masker_criterion_type=args.mask_in_criterion,
        masker_criterion_config=args.mask_in_criterion_config,
        lambda_r=args.mask_in_lambda_r,
        lambda_tv=args.mask_in_lambda_tv,
        add_prob_layers=args.add_prob_layers,
        prob_loss_func=args.prob_loss_func,
        objective_direction=args.mask_in_objective_direction,
        objective_type=args.mask_in_objective_type,
        mask_reg_mode="mask_in",
        device=device,
    )
    mask_out_criterion = criterion.resolve_masker_criterion(
        masker_criterion_type=args.mask_out_criterion,
        masker_criterion_config=args.mask_out_criterion_config,
        lambda_r=args.mask_out_lambda_r,
        lambda_tv=args.mask_out_lambda_tv,
        add_prob_layers=args.add_prob_layers,
        prob_loss_func=args.prob_loss_func,
        objective_direction=args.mask_out_objective_direction,
        objective_type=args.mask_out_objective_type,
        mask_reg_mode="mask_out",
        device=device,
    )

    if args.need_infiller:
        infiller = archs.get_infiller(args.infiller_model).to(device).eval()
        casme_runner = core.InfillerCASMERunner(
            classifier=classifier,
            masker=masker,
            classifier_optimizer=classifier_optimizer,
            masker_optimizer=masker_optimizer,
            classifier_criterion=nn.CrossEntropyLoss(),
            mask_in_criterion=mask_in_criterion,
            mask_out_criterion=mask_out_criterion,
            fixed_classifier=args.fixed_classifier,
            perc_of_training=args.perc_of_training,
            prob_historic=args.prob_historic,
            save_freq=args.save_freq,
            zoo_size=args.f_size,
            image_normalization_mode=None,
            add_prob_layers=args.add_prob_layers,
            prob_sample_low=args.prob_sample_low,
            prob_sample_high=args.prob_sample_high,
            mask_in_weight=args.mask_in_weight,
            mask_out_weight=args.mask_out_weight,
            add_class_ids=args.add_class_ids,
            device=device,
            logger=zlog.ZBufferedLogger(
                fol_path=args.log_path,
                buffer_size_dict={"messages": 1},
                default_buffer_size=1,
            ),
            infiller=infiller,
            train_infiller=False,
            do_infill_for_mask_in=args.do_infill_for_mask_in,
            do_infill_for_mask_out=args.do_infill_for_mask_out,
        )
    else:
        casme_runner = core.CASMERunner(
            classifier=classifier,
            masker=masker,
            classifier_optimizer=classifier_optimizer,
            masker_optimizer=masker_optimizer,
            classifier_criterion=nn.CrossEntropyLoss(),
            mask_in_criterion=mask_in_criterion,
            mask_out_criterion=mask_out_criterion,
            fixed_classifier=args.fixed_classifier,
            perc_of_training=args.perc_of_training,
            prob_historic=args.prob_historic,
            save_freq=args.save_freq,
            zoo_size=args.f_size,
            image_normalization_mode=None,
            add_prob_layers=args.add_prob_layers,
            prob_sample_low=args.prob_sample_low,
            prob_sample_high=args.prob_sample_high,
            mask_in_weight=args.mask_in_weight,
            mask_out_weight=args.mask_out_weight,
            add_class_ids=args.add_class_ids,
            device=device,
            logger=zlog.ZBufferedLogger(
                fol_path=args.log_path,
                buffer_size_dict={"messages": 1},
                default_buffer_size=1,
            )
        )

    # training loop
    for epoch in tqdm.trange(args.epochs, desc="Epochs"):
        single_adjust_learning_rate(
            optimizer=classifier_optimizer,
            epoch=epoch, lr=args.lr, lrde=args.lrde,
        )
        single_adjust_learning_rate(
            optimizer=masker_optimizer,
            epoch=epoch, lr=args.lr_casme, lrde=args.lrde,
        )

        # train for one epoch
        casme_runner.train_or_eval(
            data_loader=train_loader,
            is_train=True,
            epoch=epoch,
        )

        if args.do_val:
            # evaluate on validation set
            casme_runner.train_or_eval(
                data_loader=val_loader,
                is_train=False,
                epoch=epoch
            )

        # save checkpoint
        if (epoch + 1) % args.actual_save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_classifier': classifier.state_dict(),
                'state_dict_masker': masker.state_dict(),
                'optimizer_classifier': classifier_optimizer.state_dict(),
                'optimizer_masker': masker_optimizer.state_dict(),
                'args': args.to_dict(),
            }, args)


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
