from casme import stats


class LogContainers:
    def __init__(self):
        self.batch_time = stats.TimeMeter()
        self.data_time = stats.TimeMeter()
        self.losses = stats.AverageMeter()
        self.acc = stats.AverageMeter()
        self.losses_m = stats.AverageMeter()
        self.acc_m = stats.AverageMeter()
        self.statistics = stats.StatisticsContainer()
        self.masker_total_loss = stats.AverageMeter()

        self.masker_loss = stats.AverageMeter()
        self.masker_reg = stats.AverageMeter()
        self.correct_on_clean = stats.AverageMeter()
        self.mistaken_on_masked = stats.AverageMeter()
        self.nontrivially_confused = stats.AverageMeter()


class InfillingLogContainers(LogContainers):
    def __init__(self):
        super().__init__()
        self.infiller_total_loss = stats.AverageMeter()
        self.infiller_loss = stats.AverageMeter()
        self.infiller_reg = stats.AverageMeter()
        self.infiller_hole = stats.AverageMeter()
        self.infiller_valid = stats.AverageMeter()
        self.infiller_perceptual = stats.AverageMeter()
        self.infiller_style_out = stats.AverageMeter()
        self.infiller_style_comp = stats.AverageMeter()
        self.infiller_tv = stats.AverageMeter()


class GANLogContainers:
    def __init__(self):
        self.batch_time = stats.TimeMeter()
        self.data_time = stats.TimeMeter()
        self.statistics = stats.StatisticsContainer()

        self.d_loss_real = stats.HistoricalMeter()
        self.d_loss_fake = stats.HistoricalMeter()
        self.d_loss = stats.HistoricalMeter()
        self.i_loss = stats.HistoricalMeter()
