import os

import torch

import pyutils.io as io


def accuracy(output, target, topk=(1,)):
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


def adjust_learning_rate(classifier_optimizer, masker_optimizer, epoch, lr, lr_casme, lrde):
    print("DEPRECATED (adjust_learning_rate)")
    single_adjust_learning_rate(classifier_optimizer, epoch, lr, lrde)
    single_adjust_learning_rate(masker_optimizer, epoch, lr_casme, lrde)


def single_adjust_learning_rate(optimizer, epoch, lr, lrde):
    if lrde == 0 or lrde is None:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * (0.1 ** (epoch // lrde))


def save_checkpoint(state, args):
    filename = os.path.join(args.casms_path, 'epoch_{:03d}.chk'.format(state["epoch"]))
    torch.save(state, filename)


def set_args(args):

    os.makedirs(args.output_path, exist_ok=True)
    args.casms_path = os.path.join(args.output_path, args.name)
    os.makedirs(args.casms_path, exist_ok=True)
    args.log_path = os.path.join(args.casms_path, 'log')

    if args.reproduce != '':
        set_reproduction(args)

    string_args = ''
    for name in sorted(vars(args)):
        string_args += name + '=' + str(getattr(args, name)) + ', '

    io.write_json(args.to_dict(), os.path.join(args.casms_path, "args.json"))


def set_reproduction(args):
    if args.reproduce == 'F':
        args.fixed_classifier = True
        args.prob_historic = 0.0
        args.save_freq = 10000000
        if args.adversarial:
            args.lambda_r = 9
        else:
            args.lambda_r = 2.5

    if args.reproduce == 'L':
        args.fixed_classifier = False
        args.prob_historic = 0.0
        args.save_freq = 10000000
        if args.adversarial:
            args.lambda_r = 18
        else:
            args.lambda_r = 14

    if args.reproduce == 'FL':
        args.fixed_classifier = False
        args.prob_historic = 0.5
        args.save_freq = 10000000
        if args.adversarial:
            args.lambda_r = 11
        else:
            args.lambda_r = 7.5

    if args.reproduce == 'L100':
        args.fixed_classifier = False
        args.prob_historic = 0.5
        args.save_freq = 100
        if args.adversarial:
            args.lambda_r = 17
        else:
            args.lambda_r = 10

    if args.reproduce == 'L1000':
        args.fixed_classifier = False
        args.prob_historic = 0.5
        args.save_freq = 1000
        if args.adversarial:
            args.lambda_r = 17
        else:
            args.lambda_r = 10
    raise Exception("For Reference Only")
