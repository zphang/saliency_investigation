import copy
import random
import tqdm

import torch

from casme import criterion
from casme.casme_utils import per_image_normalization
from casme.train_utils import accuracy
from casme.stats import mask_to_stats

import zproto.zlogv1 as zlog


def ignore_superclass_signature(f):
    return f


class LogData:
    def __init__(self, data=None):
        self.data = data if data else {}

    def add_dict(self, d):
        for k, v in d.items():
            assert k not in self.data
            self.data[k] = v

    def __setitem__(self, key, value):
        self.data[key] = value

    def to_dict(self):
        return self.data


class BaseRunner:
    def train_or_eval(self, *args, **kwargs):
        raise NotImplementedError()

    def train_or_eval_batch(self, *args, **kwargs):
        raise NotImplementedError()

    def set_models_mode(self, is_train):
        raise NotImplementedError()


class CASMERunner(BaseRunner):
    def __init__(self,
                 classifier, masker,
                 classifier_optimizer, masker_optimizer,
                 classifier_criterion, mask_in_criterion, mask_out_criterion,
                 fixed_classifier, perc_of_training, prob_historic, save_freq, zoo_size,
                 image_normalization_mode,
                 add_prob_layers, prob_sample_low, prob_sample_high,
                 mask_in_weight,
                 mask_out_weight,
                 add_class_ids,
                 device,
                 logger: zlog.BaseZLogger = zlog.PRINT_LOGGER,
                 ):
        self.classifier = classifier
        self.classifier_for_mask = None
        self.masker = masker
        self.classifier_optimizer = classifier_optimizer
        self.masker_optimizer = masker_optimizer
        self.classifier_criterion = classifier_criterion.to(device)
        self.mask_in_criterion = mask_in_criterion
        self.mask_out_criterion = mask_out_criterion

        self.fixed_classifier = fixed_classifier
        self.perc_of_training = perc_of_training
        self.prob_historic = prob_historic
        self.save_freq = save_freq
        self.zoo_size = zoo_size
        self.image_normalization_mode = image_normalization_mode

        self.add_prob_layers = add_prob_layers
        self.prob_sample_low = prob_sample_low
        self.prob_sample_high = prob_sample_high
        self.mask_in_weight = mask_in_weight
        self.mask_out_weight = mask_out_weight
        self.add_class_ids = add_class_ids

        self.device = device
        self.logger = logger

        self.do_mask_in = self.mask_in_criterion is not None
        self.do_mask_out = self.mask_out_criterion is not None
        self.classifier_zoo = {}
        assert self.do_mask_in or self.do_mask_out

    def train_or_eval(self, data_loader, is_train=False, epoch=None):
        self.set_models_mode(is_train)
        length = len(data_loader) * self.perc_of_training if is_train else len(data_loader)
        desc = "Train" if is_train else "Eval"
        for i, (x, y) in tqdm.tqdm(enumerate(data_loader), total=length, desc=desc):
            if i > length:
                break
            x, y = x.to(self.device), y.to(self.device)
            x = per_image_normalization(x, mode=self.image_normalization_mode)
            self.train_or_eval_batch(x=x, y=y, i=i, is_train=is_train)
            if is_train:
                self.logger.write_entry("train_status", {
                    "epoch": epoch,
                    "i": i,
                    "epoch_t": length,
                    "dataset_t": len(data_loader),
                })
            else:
                self.logger.write_entry("val_status", {
                    "i": i,
                    "dataset_t": len(data_loader),
                })

    def train_or_eval_batch(self, x, y, i, epoch=None, is_train=False):
        log_data = LogData({"epoch": epoch, "i": i})

        use_p = self.maybe_add_prob_layers(x)

        y_hat, classifier_loss, layers = self.compute_classifier_loss(
            x=x, y=y, log_data=log_data, is_train=is_train,
        )
        self.maybe_update_classifier(
            classifier_loss=classifier_loss, i=i, is_train=is_train,
        )

        with torch.set_grad_enabled(is_train):
            # compute mask and masked input
            mask = self.masker(
                layers=self.detach_layers(layers),
                use_p=use_p,
                class_ids=y if self.add_class_ids else None,
            )
            classifier_for_mask, update_classifier = self.choose_masked_classifier(is_train)
            log_data.add_dict(mask_to_stats(mask))

            classifier_loss_from_masked_in_x, mask_in_loss = self.compute_loss_for_mask_in(
                x=x, y=y, y_hat=y_hat, mask=mask, log_data=log_data, use_p=use_p,
            )

            classifier_loss_from_masked_out_x, mask_out_loss = self.compute_loss_for_mask_out(
                x=x, y=y, y_hat=y_hat, mask=mask, log_data=log_data, use_p=use_p,
            )
            classifier_loss_from_masked_x, masker_total_loss = self.aggregate_losses(
                classifier_loss_from_masked_in_x=classifier_loss_from_masked_in_x,
                mask_in_loss=mask_in_loss,
                classifier_loss_from_masked_out_x=classifier_loss_from_masked_out_x,
                mask_out_loss=mask_out_loss,
                log_data=log_data,
            )

        self.maybe_update_masked_classifier_and_masker(
            classifier_loss_from_masked_x=classifier_loss_from_masked_x, masker_total_loss=masker_total_loss,
            is_train=is_train, update_classifier=update_classifier,
        )

        self.logger.write_entry(
            "{}_batch_stats".format("train" if is_train else "val"), log_data.to_dict()
        )

    def set_models_mode(self, is_train):
        if is_train:
            self.masker.train()
            if self.fixed_classifier:
                self.classifier.eval()
            else:
                self.classifier.train()
        else:
            self.masker.eval()
            self.classifier.eval()

    def maybe_save_classifier_to_history(self, i):
        # save classifier (needed only if previous iterations are used i.e. args.hp > 0)
        if self.prob_historic > 0 \
                and ((i % self.save_freq == -1 % self.save_freq) or len(self.classifier_zoo) < 1):
            self.logger.write_entry("messages", 'Current iteration is saving, will be used in the future. ')
            if len(self.classifier_zoo) < self.zoo_size:
                index = len(self.classifier_zoo)
            else:
                index = random.randint(0, len(self.classifier_zoo) - 1)
            state_dict = self.classifier.state_dict()
            self.classifier_zoo[index] = {}
            for p in state_dict:
                self.classifier_zoo[index][p] = state_dict[p].cpu()
            self.logger.write_entry("messages", 'There are {0} iterations stored.'.format(len(self.classifier_zoo)))

    def setup_classifier_for_mask(self):
        if self.classifier_for_mask is None:
            self.classifier_for_mask = copy.deepcopy(self.classifier)
        index = random.randint(0, len(self.classifier_zoo) - 1)
        self.classifier_for_mask.load_state_dict(self.classifier_zoo[index])
        self.classifier_for_mask.eval()

    def classifier_optimizer_step(self, classifier_loss):
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

    def classifier_from_masked_optimizer_step(self, classifier_loss_from_masked_x):
        self.classifier_optimizer.zero_grad()
        classifier_loss_from_masked_x.backward(retain_graph=True)
        self.classifier_optimizer.step()

    def masker_optimizer_step(self, masker_total_loss):
        self.masker_optimizer.zero_grad()
        masker_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.masker.parameters(), 10)
        self.masker_optimizer.step()

    def choose_masked_classifier(self, is_train):
        if (not is_train) or self.fixed_classifier or (random.random() > self.prob_historic):
            classifier_for_mask = self.classifier
            update_classifier = not self.fixed_classifier
        else:
            self.setup_classifier_for_mask()
            classifier_for_mask = self.classifier_for_mask
            update_classifier = False
        return classifier_for_mask, update_classifier

    def maybe_add_prob_layers(self, x):
        if self.add_prob_layers:
            use_p = torch.Tensor(x.shape[0])\
                .uniform_(self.prob_sample_low, self.prob_sample_high)\
                .to(self.device)
        else:
            use_p = None
        return use_p

    def compute_classifier_loss(self, x, y, log_data, is_train):
        # compute classifier prediction on the original images and get inner layers
        with torch.set_grad_enabled(is_train and (not self.fixed_classifier)):
            y_hat, layers = self.classifier(x, return_intermediate=True)
            classifier_loss = self.classifier_criterion(y_hat, y)
            log_data["classifier_loss"] = classifier_loss.item()
            log_data["acc"] = accuracy(y_hat.detach(), y, topk=(1,))[0].item()
        return y_hat, classifier_loss, layers

    def maybe_update_classifier(self, classifier_loss, i, is_train):
        # update classifier - compute gradient and do SGD step for clean image, save classifier
        if is_train and (not self.fixed_classifier):
            self.classifier_optimizer_step(classifier_loss=classifier_loss)
            self.maybe_save_classifier_to_history(i=i)

    @classmethod
    def mask_in_x(cls, x, mask):
        masked_in_x = criterion.MaskFunc.mask_in(x=x, mask=mask)
        return masked_in_x

    def compute_loss_for_mask_in(self, x, y, y_hat, mask, log_data, use_p=None):
        if self.do_mask_in:
            masked_in_x = self.mask_in_x(x=x, mask=mask)
            y_hat_from_masked_in_x = self.classifier(masked_in_x)
            classifier_loss_from_masked_in_x = self.classifier_criterion(y_hat_from_masked_in_x, y)
            mask_in_loss, mask_in_loss_metadata = self.mask_in_criterion(
                mask=mask, y_hat=y_hat, y_hat_from_masked_x=y_hat_from_masked_in_x, y=y,
                classifier_loss_from_masked_x=classifier_loss_from_masked_in_x, use_p=use_p,
            )
            log_data["masked_in___classifier_loss"] = classifier_loss_from_masked_in_x.item()
            log_data["masked_in___acc"] = accuracy(y_hat_from_masked_in_x.detach(), y, topk=(1,))[0].item()
            log_data["masked_in___correct_on_masked"] = mask_in_loss_metadata["correct_on_masked"].item()
            log_data["masked_in___mask_reg"] = mask_in_loss_metadata["mask_reg"].item()
        else:
            classifier_loss_from_masked_in_x = mask_in_loss = 0
        return classifier_loss_from_masked_in_x, mask_in_loss

    @classmethod
    def mask_out_x(cls, x, mask):
        masked_out_x = criterion.MaskFunc.mask_out(x=x, mask=mask)
        return masked_out_x

    def compute_loss_for_mask_out(self, x, y, y_hat, mask, log_data, use_p=None):
        if self.do_mask_out:
            masked_out_x = self.mask_out_x(x=x, mask=mask)
            y_hat_from_masked_out_x = self.classifier(masked_out_x)
            classifier_loss_from_masked_out_x = self.classifier_criterion(y_hat_from_masked_out_x, y)
            mask_out_loss, mask_out_loss_metadata = self.mask_out_criterion(
                mask=mask, y_hat=y_hat, y_hat_from_masked_x=y_hat_from_masked_out_x, y=y,
                classifier_loss_from_masked_x=classifier_loss_from_masked_out_x, use_p=use_p,
            )
            log_data["masked_out___classifier_loss"] = classifier_loss_from_masked_out_x.item()
            log_data["masked_out___acc"] = accuracy(y_hat_from_masked_out_x.detach(), y, topk=(1,))[0].item()
            log_data["masked_out___mistaken_on_masked"] = mask_out_loss_metadata["mistaken_on_masked"].item()
            log_data["masked_out___nontrivially_confused"] = mask_out_loss_metadata["nontrivially_confused"].item()
            log_data["masked_out___mask_reg"] = mask_out_loss_metadata["mask_reg"].item()
        else:
            classifier_loss_from_masked_out_x = mask_out_loss = 0
        return classifier_loss_from_masked_out_x, mask_out_loss

    def aggregate_losses(self,
                         classifier_loss_from_masked_in_x, mask_in_loss,
                         classifier_loss_from_masked_out_x, mask_out_loss,
                         log_data
                         ):
        classifier_loss_from_masked_x = (
            self.mask_in_weight * classifier_loss_from_masked_in_x
            + self.mask_out_weight * classifier_loss_from_masked_out_x
        )
        masker_total_loss = (
            self.mask_in_weight * mask_in_loss
            + self.mask_out_weight * mask_out_loss
        )

        log_data["masked_total___classifier_loss"] = classifier_loss_from_masked_x.item()
        log_data["masked_total___loss"] = masker_total_loss.item()

        return classifier_loss_from_masked_x, masker_total_loss

    def maybe_update_masked_classifier_and_masker(
            self, classifier_loss_from_masked_x, masker_total_loss,
            is_train, update_classifier):
        if is_train:
            # update classifier - compute gradient, do SGD step for masked image
            if update_classifier:
                self.classifier_from_masked_optimizer_step(classifier_loss_from_masked_x)

            # update casme - compute gradient, do SGD step
            self.masker_optimizer_step(masker_total_loss)

    @classmethod
    def detach_layers(cls, layers):
        return [layer.detach() for layer in layers]


class InfillerCASMERunner(CASMERunner):

    def __init__(self,
                 infiller,
                 # infiller_optimizer,
                 # infiller_criterion,
                 train_infiller,
                 do_infill_for_mask_in, do_infill_for_mask_out,
                 **kwargs):
        self.infiller = infiller
        # self.infiller_optimizer = infiller_optimizer
        # self.infiller_criterion = infiller_criterion
        self.train_infiller = train_infiller
        self.do_infill_for_mask_in = do_infill_for_mask_in
        self.do_infill_for_mask_out = do_infill_for_mask_out
        super().__init__(**kwargs)

    def mask_in_x(self, x, mask):
        masked_in_x = super().mask_in_x(x=x, mask=mask)

        if self.do_infill_for_mask_in:
            return infill_masked_in(
                infiller=self.infiller,
                masked_in_x=masked_in_x,
                mask=mask, x=x,
                train_infiller=self.train_infiller,
            )
        else:
            return masked_in_x

    def mask_out_x(self, x, mask):
        masked_out_x = super().mask_out_x(x=x, mask=mask)

        if self.do_infill_for_mask_out:
            return infill_masked_out(
                infiller=self.infiller,
                masked_out_x=masked_out_x,
                mask=mask, x=x,
                train_infiller=self.train_infiller,
            )
        else:
            return masked_out_x

    def set_models_mode(self, is_train):
        super().set_models_mode(is_train)
        if is_train and self.train_infiller:
            self.infiller.train()
        else:
            self.infiller.eval()


def infill_masked_in(infiller, masked_in_x, mask, x, train_infiller=False):
    with torch.set_grad_enabled(train_infiller):
        generated = infiller(
            masked_x=masked_in_x, mask=mask, x=x,
            mask_mode=criterion.MaskFunc.MASK_IN,
        )
    return criterion.InfillFunc.infill_for_mask_in(
        masked_x=masked_in_x, mask=mask,
        infill_data=generated,
    )


def infill_masked_out(infiller, masked_out_x, mask, x, train_infiller=False):
    with torch.set_grad_enabled(train_infiller):
        generated = infiller(
            masked_x=masked_out_x, mask=mask, x=x,
            mask_mode=criterion.MaskFunc.MASK_OUT,
        )
    return criterion.InfillFunc.infill_for_mask_out(
        masked_x=masked_out_x, mask=mask,
        infill_data=generated,
    )
