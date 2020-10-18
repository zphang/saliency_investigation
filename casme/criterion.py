import json

import torch
import torch.nn as nn
import torch.nn.functional as F


from pyutils.datastructures import combine_dicts


class MaskFunc:

    MASK_IN = "MASK_IN"
    MASK_OUT = "MASK_OUT"
    MODES = [MASK_IN, MASK_OUT]

    def __init__(self, mask_mode):
        assert mask_mode in self.MODES
        self.mask_mode = mask_mode

    def apply_mask(self, x, mask):
        return self.static_apply(
            x=x, mask=mask, mask_mode=self.mask_mode,
        )

    @classmethod
    def static_apply(cls, x, mask, mask_mode):
        if mask_mode == cls.MASK_IN:
            return cls.mask_in(x, mask)
        elif mask_mode == cls.MASK_OUT:
            return cls.mask_out(x, mask)
        else:
            raise KeyError(mask_mode)

    @staticmethod
    def mask_out(x, mask):
        return x * (1 - mask)

    @staticmethod
    def mask_in(x, mask):
        return x * mask


class InfillFunc:

    @staticmethod
    def infill_for_mask_out(masked_x, mask, infill_data):
        # re-masking
        # return masked_x * (1 - mask) + infilled_x * mask
        # without re-masking
        return masked_x + infill_data * mask

    @staticmethod
    def infill_for_mask_in(masked_x, mask, infill_data):
        # re-masking
        # return masked_x * mask + infilled_x * (1 - mask)
        # without re-masking
        return masked_x + infill_data * (1 - mask)


def apply_uniform_random_value_mask_func(x, mask):
    # for mask, return image infilled with random value between 0 and 1?
    # zero_mean_std_1?
    # x[mask==1] = np.random.random((mask==1).shape)
    return None


def default_infill_func(x, mask, generated_image):
    return generated_image * mask + x * (1-mask)


class MaskerCriterion(nn.Module):
    def __init__(self, lambda_r, add_prob_layers, prob_loss_func,
                 objective_direction, objective_type,
                 mask_reg_mode,
                 device, y_hat_log_softmax=False, lambda_tv=None):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.objective_direction = objective_direction
        self.objective_type = objective_type
        self.mask_reg_mode = mask_reg_mode
        self.device = device
        self.y_hat_log_softmax = y_hat_log_softmax
        self.lambda_tv = lambda_tv

        assert mask_reg_mode in ["mask_in", "mask_out"]

    def forward(self,
                mask, y_hat, y_hat_from_masked_x, y,
                classifier_loss_from_masked_x, use_p, reduce=True):
        _, max_indexes = y_hat.detach().max(1)
        _, max_indexes_on_masked_x = y_hat_from_masked_x.detach().max(1)
        correct_on_clean = y.eq(max_indexes).long()
        metadata = {}

        mask_mean = F.avg_pool2d(mask, 224, stride=1).squeeze()
        metadata["mask_mean"] = mask_mean
        # Potentially rename to mask_size or mask_size_for_reg
        if self.add_prob_layers:
            metadata["use_p"] = use_p
            # adjust to minimize deviation from p
            mask_mean = (mask_mean - use_p)
            if self.prob_loss_func == "l1":
                mask_mean = mask_mean.abs()
            elif self.prob_loss_func == "l2":
                mask_mean = mask_mean.pow(2)
            else:
                raise KeyError(self.prob_loss_func)

        # apply regularization loss only on non-trivially confused images
        mask_reg, mask_reg_metadata = self.compute_mask_regularization(
            y=y, max_indexes_on_masked_x=max_indexes_on_masked_x,
            correct_on_clean=correct_on_clean, mask_mean=mask_mean,
            reduce=reduce,
        )
        tv_reg = tv_loss(mask=mask, tv_weight=self.lambda_tv)
        regularization = mask_reg + tv_reg

        loss, loss_metadata = self.compute_only_loss(
            y_hat_from_masked_x=y_hat_from_masked_x,
            y=y,
            classifier_loss_from_masked_x=classifier_loss_from_masked_x,
            correct_on_clean=correct_on_clean,
            reduce=reduce,
        )
        masker_loss = loss + regularization

        if reduce:
            metadata["regularization"] = regularization.mean()
            metadata["correct_on_clean"] = correct_on_clean.float().mean()

        else:
            metadata["regularization"] = regularization
            metadata["correct_on_clean"] = correct_on_clean.float()

        metadata["loss"] = loss
        metadata["tv_reg"] = tv_reg

        metadata = combine_dicts([metadata, mask_reg_metadata], strict=True)
        if self.objective_type == "entropy":
            metadata["negative_entropy"] = loss_metadata["negative_entropy"]
        return masker_loss, metadata

    def compute_only_loss(self, y_hat_from_masked_x, y, classifier_loss_from_masked_x,
                          correct_on_clean, reduce):

        loss_metadata = {}
        # main loss for casme
        if self.objective_type == "classification":
            if reduce:
                objective = - classifier_loss_from_masked_x
            else:
                # This is a hack.
                if self.y_hat_log_softmax:
                    classifier_criterion = F.nll_loss
                else:
                    classifier_criterion = F.cross_entropy
                objective = - classifier_criterion(y_hat_from_masked_x, y, reduction='none')
        elif self.objective_type == "entropy":
            if self.y_hat_log_softmax:
                log_prob = y_hat_from_masked_x
            else:
                log_prob = F.log_softmax(y_hat_from_masked_x, dim=-1)
            prob = log_prob.exp()
            entropy = -(log_prob * prob).sum(1)
            # apply main loss only when original images are correctly classified
            entropy_correct = entropy * correct_on_clean.float()
            objective = entropy_correct
            loss_metadata["negative_entropy"] = -entropy
        else:
            raise KeyError(self.objective_type)

        if reduce:
            objective = objective.mean()

        loss = self.determine_loss_direction(objective)
        return loss, loss_metadata

    def determine_loss_direction(self, loss):
        if self.objective_direction == "maximize":
            # maximize objective = minimize negative loss
            return -loss
        elif self.objective_direction == "minimize":
            # minimize objective = minimize loss
            return loss
        else:
            raise KeyError(self.loss_type)

    def compute_mask_out_regularization(self, y, max_indexes_on_masked_x, correct_on_clean, mask_mean,
                                        reduce):
        """
        nontrivially_confused = 1 if correct on clean, wrong in masked
        mask_mean = mask% for that image
        F.relu(nontrivially_confused - mask_mean) > 0 only if nontrivially confused

        small mask -> large F.relu -> more negative
        """
        mistaken_on_masked = y.ne(max_indexes_on_masked_x).long()
        nontrivially_confused = (correct_on_clean + mistaken_on_masked).eq(2).float()
        mask_reg = -self.lambda_r * F.relu(nontrivially_confused - mask_mean)
        mask_reg_mean = mask_reg.mean()
        if reduce:
            mask_reg_metadata = {
                "mistaken_on_masked": mistaken_on_masked.float().mean(),
                "nontrivially_confused": nontrivially_confused.mean(),
                "mask_reg": mask_reg_mean,
            }
        else:
            mask_reg_metadata = {
                "mistaken_on_masked": mistaken_on_masked.float(),
                "nontrivially_confused": nontrivially_confused,
                "mask_reg": mask_reg,
            }
        return mask_reg_mean, mask_reg_metadata

    def compute_mask_in_regularization(self, y, max_indexes_on_masked_x, mask_mean, reduce):
        """
         F.relu(correct_on_masked - mask_mean) > 0 only if correct on masked
        """
        correct_on_masked = y.eq(max_indexes_on_masked_x).float()
        mask_reg = -self.lambda_r * F.relu(correct_on_masked - mask_mean)
        mask_reg_mean = mask_reg.mean()
        if reduce:
            mask_reg_metadata = {
                "correct_on_masked": correct_on_masked.mean(),
                "mask_reg": mask_reg_mean,
            }
        else:
            mask_reg_metadata = {
                "correct_on_masked": correct_on_masked,
                "mask_reg": mask_reg,
            }
        return mask_reg_mean, mask_reg_metadata

    def compute_mask_regularization(self, y, max_indexes_on_masked_x, correct_on_clean, mask_mean,
                                    use_p=None, reduce=True):
        if self.mask_reg_mode == "mask_out":
            return self.compute_mask_out_regularization(
                y=y,
                max_indexes_on_masked_x=max_indexes_on_masked_x,
                correct_on_clean=correct_on_clean,
                mask_mean=mask_mean,
                reduce=reduce,
            )
        elif self.mask_reg_mode == "mask_in":
            return self.compute_mask_in_regularization(
                y=y,
                max_indexes_on_masked_x=max_indexes_on_masked_x,
                mask_mean=mask_mean,
                reduce=reduce,
            )
        else:
            raise KeyError(self.mask_reg_mode)


class MaskerPriorCriterion(nn.Module):
    def __init__(self, lambda_r, class_weights, add_prob_layers, prob_loss_func, config, device,
                 y_hat_log_softmax=False):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.config = json.loads(config)
        self.device = device
        self.y_hat_log_softmax = y_hat_log_softmax

        self.class_weights = torch.Tensor(class_weights).to(device)
        inverse_class_weights = 1 / self.class_weights
        self.prior = (inverse_class_weights / inverse_class_weights.sum())

    def forward(self,
                mask, y_hat, y_hat_from_masked_x, y,
                classifier_loss_from_masked_x, use_p, reduce=True):
        if self.y_hat_log_softmax:
            y_hat_prob = torch.exp(y_hat)
            y_hat_from_masked_x_prob = torch.exp(y_hat_from_masked_x)
        else:
            y_hat_prob = F.softmax(y_hat, dim=-1)
            y_hat_from_masked_x_prob = F.softmax(y_hat_from_masked_x, dim=-1)

        # Should this be / or - ?
        if self.config["prior"] == "subtract":
            y_hat_is_over_prior = y_hat_prob - self.prior
            y_hat_from_masked_x_prob_over_prior = y_hat_from_masked_x_prob - self.prior
        elif self.config["prior"] == "divide":
            y_hat_is_over_prior = y_hat_prob / self.prior
            y_hat_from_masked_x_prob_over_prior = y_hat_from_masked_x_prob / self.prior
        else:
            raise KeyError(self.config["prior"])

        _, max_indexes = y_hat_is_over_prior.detach().max(1)
        _, max_indexes_on_masked_x = y_hat_from_masked_x_prob_over_prior.detach().max(1)

        correct_on_clean = y.eq(max_indexes).long()
        mistaken_on_masked = y.ne(max_indexes_on_masked_x).long()
        nontrivially_confused = (correct_on_clean + mistaken_on_masked).eq(2).float()

        mask_mean = F.avg_pool2d(mask, 224, stride=1).squeeze()
        if self.add_prob_layers:
            # adjust to minimize deviation from p
            mask_mean = (mask_mean - use_p)
            if self.prob_loss_func == "l1":
                mask_mean = mask_mean.abs()
            elif self.prob_loss_func == "l2":
                mask_mean = mask_mean.pow(2)
            else:
                raise KeyError(self.prob_loss_func)

        # apply regularization loss only on non-trivially confused images
        if self.add_prob_layers or self.config.get("regularize_always"):
            regularization = self.lambda_r * mask_mean
        else:
            regularization = -self.lambda_r * F.relu(nontrivially_confused - mask_mean)

        # main loss for casme
        if self.y_hat_log_softmax:
            log_prob = y_hat_from_masked_x
        else:
            log_prob = F.log_softmax(y_hat_from_masked_x, dim=-1)
        if self.config["kl"] == "forward":
            # - sum: p_i log(q_i)
            kl = - (self.prior * log_prob).sum(dim=-1)
        elif self.config["kl"] == "backward":
            log_prior = torch.log(self.prior)
            # - sum: q_i log(p_i / q_i)
            kl = - (y_hat_from_masked_x_prob * (log_prior - log_prob)).sum(dim=-1)
        else:
            raise KeyError(self.config["kl"])

        # apply main loss only when original images are correctly classified
        if self.config["loss_on_coc"]:
            kl = kl * correct_on_clean.float()
        else:
            kl = kl

        if "nothing_class" in self.config:
            keep_filter = (y != self.config["nothing_class"]).float()
            kl = kl * keep_filter

        if self.config["apply_class_weight"]:
            sample_weights = torch.index_select(self.class_weights, dim=0, index=y)
            regularization = regularization * sample_weights
            kl = kl * sample_weights

        if "nothing_class_reg_weight" in self.config:
            reg_weight = (
                (y == self.config["nothing_class"]).float()
                * (self.config["nothing_class_reg_weight"] - 1)
                + 1
            )
            regularization = regularization * reg_weight

        if reduce:
            regularization = regularization.mean()
            loss = kl.mean()
            masker_loss = loss + regularization
            metadata = {
                "correct_on_clean": correct_on_clean.float().mean(),
                "mistaken_on_masked": mistaken_on_masked.float().mean(),
                "nontrivially_confused": nontrivially_confused.float().mean(),
                "loss": loss,
                "regularization": regularization,
            }
        else:
            loss = kl
            masker_loss = loss + regularization
            metadata = {
                "correct_on_clean": correct_on_clean.float(),
                "mistaken_on_masked": mistaken_on_masked.float(),
                "nontrivially_confused": nontrivially_confused.float(),
                "loss": loss,
                "regularization": regularization,
            }
        return masker_loss, metadata


class MaskerInfillerPriorCriterion(nn.Module):
    def __init__(self, lambda_r, class_weights, add_prob_layers, prob_loss_func, config, device,
                 y_hat_log_softmax=False):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.config = json.loads(config)
        self.device = device
        self.y_hat_log_softmax = y_hat_log_softmax

        self.class_weights = torch.Tensor(class_weights).to(device)
        inverse_class_weights = 1 / self.class_weights
        self.prior = (inverse_class_weights / inverse_class_weights.sum())

    def forward(self,
                mask, modified_x,
                # generated_image, infilled_image,
                # layers, generated_layers, infilled_layers, dilated_boundaries,
                y_hat, y_hat_from_modified_x, y,
                classifier_loss_from_modified_x, use_p):
        y_hat_prob = F.softmax(y_hat, dim=1)
        y_hat_from_modified_x_prob = F.softmax(y_hat_from_modified_x, dim=1)

        # Should this be / or - ?
        if self.config["prior"] == "subtract":
            y_hat_is_over_prior = y_hat_prob - self.prior
            y_hat_from_modified_x_prob_over_prior = y_hat_from_modified_x - self.prior
        elif self.config["prior"] == "divide":
            y_hat_is_over_prior = y_hat_prob / self.prior
            y_hat_from_modified_x_prob_over_prior = y_hat_from_modified_x_prob / self.prior
        else:
            raise KeyError(self.config["prior"])

        _, max_indexes = y_hat_is_over_prior.detach().max(1)
        _, max_indexes_on_modified_x = y_hat_from_modified_x_prob_over_prior.detach().max(1)

        correct_on_clean = y.eq(max_indexes).long()
        mistaken_on_masked = y.ne(max_indexes_on_modified_x).long()
        nontrivially_confused = (correct_on_clean + mistaken_on_masked).eq(2).float()

        mask_mean = F.avg_pool2d(mask, 224, stride=1).squeeze()
        if self.add_prob_layers:
            # adjust to minimize deviation from p
            print("A", mask_mean.mean())
            mask_mean = (mask_mean - use_p)
            if self.prob_loss_func == "l1":
                mask_mean = mask_mean.abs()
            elif self.prob_loss_func == "l2":
                mask_mean = mask_mean.pow(2)
            else:
                raise KeyError(self.prob_loss_func)
            print("B", mask_mean.mean())

        # apply regularization loss only on non-trivially confused images
        if self.add_prob_layers:
            regularization = self.lambda_r * mask_mean
        else:
            regularization = -self.lambda_r * F.relu(nontrivially_confused - mask_mean)

        # main loss for casme
        log_prob = F.log_softmax(y_hat_from_modified_x, dim=-1)
        if self.config["kl"] == "forward":
            # - sum: p_i log(q_i)
            kl = - (self.prior * log_prob).sum(dim=-1)
        elif self.config["kl"] == "backward":
            log_prior = torch.log(self.prior)
            # - sum: q_i log(p_i / q_i)
            kl = - (y_hat_from_modified_x * (log_prior - log_prob)).sum(dim=-1)
        elif self.config["kl"] == "backward_ce":
            log_prior = torch.log(self.prior)
            # - sum: q_i log(p_i / q_i)
            kl = - (y_hat_from_modified_x * log_prior).sum(dim=-1)
        else:
            raise KeyError(self.config["kl"])

        # apply main loss only when original images are correctly classified
        if self.config["loss_on_coc"]:
            kl = kl * correct_on_clean.float()
        else:
            kl = kl

        if "nothing_class" in self.config:
            keep_filter = (y != self.config["nothing_class"]).float()
            kl = kl * keep_filter

        if self.config["apply_class_weight"]:
            sample_weights = torch.index_select(self.class_weights, dim=0, index=y)
            regularization = regularization * sample_weights
            kl = kl * sample_weights

        if "nothing_class_reg_weight" in self.config:
            reg_weight = (
                (y == self.config["nothing_class"]).float()
                * (self.config["nothing_class_reg_weight"] - 1)
                + 1
            )
            regularization = regularization * reg_weight

        regularization = regularization.mean()
        loss = kl.mean()

        masker_loss = loss + regularization
        print("C", regularization.item())
        metadata = {
            "correct_on_clean": correct_on_clean.float().mean(),
            "mistaken_on_masked": mistaken_on_masked.float().mean(),
            "nontrivially_confused": nontrivially_confused.float().mean(),
            "loss": loss,
            "regularization": regularization,
        }
        return masker_loss, metadata


def resolve_masker_criterion(masker_criterion_type, masker_criterion_config,
                             mask_reg_mode,
                             lambda_r, lambda_tv,
                             add_prob_layers, prob_loss_func,
                             objective_direction, objective_type,
                             device):
    if masker_criterion_type == "crossentropy":
        masker_criterion = MaskerCriterion(
            lambda_r=lambda_r,
            lambda_tv=lambda_tv,
            add_prob_layers=add_prob_layers,
            prob_loss_func=prob_loss_func,
            objective_direction=objective_direction,
            objective_type=objective_type,
            mask_reg_mode=mask_reg_mode,
            device=device,
        ).to(device)
    elif masker_criterion_type == "kldivergence":
        masker_criterion = MaskerPriorCriterion(
            lambda_r=lambda_r,
            class_weights=[1 / 1000] * 1000,
            add_prob_layers=add_prob_layers,
            prob_loss_func=prob_loss_func,
            config=masker_criterion_config,
            device=device,
        ).to(device)
    elif masker_criterion_type == "none":
        masker_criterion = None
    else:
        raise KeyError(masker_criterion_type)
    return masker_criterion


class DiscriminatorCriterion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, real_images_logits, gen_images_logits):
        batch_size = real_images_logits.shape[0]  # TODO

        valid = torch.tensor([1.] * batch_size, device=self.device, requires_grad=False, dtype=torch.float32).view(-1,
                                                                                                                   1)
        fake = torch.tensor([0.] * batch_size, device=self.device, requires_grad=False, dtype=torch.float32).view(-1, 1)
        generator_loss = self.adversarial_loss(gen_images_logits, valid)
        real_loss = self.adversarial_loss(real_images_logits, valid)
        fake_loss = self.adversarial_loss(gen_images_logits, fake)
        metadata = {
            'generator_loss': generator_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
        }
        discriminator_loss = (real_loss + fake_loss) / 2
        return generator_loss, discriminator_loss, metadata


def determine_mask_func(objective_direction, objective_type):
    if objective_type == "classification" and objective_direction == "maximize":
        mask_func = MaskFunc(mask_mode=MaskFunc.MASK_IN)
    elif objective_type == "classification" and objective_direction == "minimize":
        mask_func = MaskFunc(mask_mode=MaskFunc.MASK_OUT)
    elif objective_type == "entropy" and objective_direction == "maximize":
        mask_func = MaskFunc(mask_mode=MaskFunc.MASK_OUT)
    elif objective_type == "entropy" and objective_direction == "minimize":
        mask_func = MaskFunc(mask_mode=MaskFunc.MASK_IN)
    else:
        raise KeyError((objective_type, objective_direction))
    return mask_func


def tv_loss(mask, tv_weight, power=2, border_penalty=0.3):
    if tv_weight is None or tv_weight == 0:
        return 0.0
    # https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    # https://github.com/PiotrDabkowski/pytorch-saliency/blob/bfd501ec7888dbb3727494d06c71449df1530196/sal/utils/mask.py#L5
    w_variance = torch.sum(torch.pow(mask[:, :, :, :-1] - mask[:, :, :, 1:], power))
    h_variance = torch.sum(torch.pow(mask[:, :, :-1, :] - mask[:, :, 1:, :], power))
    if border_penalty > 0:
        border = float(border_penalty)*torch.sum(
            mask[:, :, -1, :]**power + mask[:, :, 0, :]**power
            + mask[:, :, :, -1]**power + mask[:, :, :, 0]**power
        )
    else:
        border = 0.
    loss = tv_weight * (h_variance + w_variance + border) / float(power * mask.size(0))
    return loss

