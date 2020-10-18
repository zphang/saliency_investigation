import numpy as np
import time

import torch


class AverageMeter:
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter:
    def __init__(self):
        self.meter = AverageMeter()
        self.time_mark = None
        self.reset()

    @property
    def val(self):
        return self.meter.val

    @property
    def avg(self):
        return self.meter.avg

    @property
    def sum(self):
        return self.meter.sum

    @property
    def count(self):
        return self.meter.count

    def reset(self):
        self.time_mark = time.time()

    def update(self):
        self.meter.update(time.time() - self.time_mark)
        self.time_mark = time.time()


def mask_to_stats(mask):
    mask_stats = {}
    with torch.no_grad():
        mask = mask.detach()

        mask_stats["mask_avg"] = mask.mean().item()
        mask_stats["mask_std"] = mask.mean(3).mean(2).std().item()

        flat = mask.view(-1).cpu().numpy()
        non_zero_flat = flat[flat > 0]
        clear_flat = non_zero_flat[non_zero_flat < 1]
        clear_flat_log2 = np.log2(clear_flat)
        sum_across_batch = -np.sum(clear_flat * clear_flat_log2)
        mask_stats["mask_entropy"] = sum_across_batch / flat.size

        tv = (
                (mask[:, :, :, :-1] - mask[:, :, :, 1:]).pow(2).mean()
                + (mask[:, :, :-1, :] - mask[:, :, 1:, :]).pow(2).mean()
        )
        mask_stats["mask_tv"] = tv.item()
    return mask_stats


class StatisticsContainer:
    def __init__(self):
        self.avg = None
        self.std = None
        self.entropy = None
        self.tv = None
        self.reset()

    def reset(self):
        self.avg = AverageMeter()
        self.std = AverageMeter()
        self.entropy = AverageMeter()
        self.tv = AverageMeter()

    def update(self, mask):
        with torch.no_grad():
            mask = mask.detach()

            self.avg.update(mask.mean().item(), mask.size(0))

            self.std.update(mask.mean(3).mean(2).std().item(), mask.size(0))

            flat = mask.view(-1).cpu().numpy()
            non_zero_flat = flat[flat > 0]
            clear_flat = non_zero_flat[non_zero_flat < 1]
            clear_flat_log2 = np.log2(clear_flat)
            sum_across_batch = -np.sum(clear_flat*clear_flat_log2)
            self.entropy.update(sum_across_batch/flat.size, mask.size(0))

            tv = (
                (mask[:, :, :, :-1] - mask[:, :, :, 1:]).pow(2).mean()
                + (mask[:, :, :-1, :] - mask[:, :, 1:, :]).pow(2).mean()
            )
            self.tv.update(tv.item(), mask.size(0))

    def str_out(self):
        return (
            'TV (x100)   {tv_avg:.3f} ({tv_val:.3f})\t'
            'AvgMask {a.avg:.3f} ({a.val:.3f})\n'
            'EntropyMask {e.avg:.3f} ({e.val:.3f})\t'
            'StdMask {s.avg:.3f} ({s.val:.3f})'.format(
                a=self.avg, s=self.std, e=self.entropy, tv_avg=100*self.tv.avg,
                tv_val=100*self.tv.val
            ))

    def print_out(self):
        print(self.str_out(), flush=True)

    def get_dictionary(self):
        return {
            'avg_mask': self.avg.avg,
            'std_mask': self.std.avg,
            'entropy': self.entropy.avg,
            'tv': self.tv.avg,
        }


class HistoricalMeter:
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.history = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)
