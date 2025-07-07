from pathlib import Path


def get_log_name(*args):
    name = args[0]
    for i in range(1, len(args)):
        name = name + "_" + str(args[i])
    return name


class ConfigTrain:
    def __init__(
        self,
        n_gpus,
        model_type="L",
        foundation_models="DA_DepthPro",
        checkpoint=None,
    ):
        super().__init__()
        # data configs
        self.data_dirs = Path("Datasets")  # data path
        self.checkpoint = None  # checkpoint path

        # data configs
        self.sizes = 320  # sizes of images during training
        self.foundation_models = foundation_models  # pseudo data configs

        # optimizer setting
        self.lr = 2e-4  # learning rate
        self.wd = 0.05  # weight decay
        self.epochs = 100  # epochs numbers
        self.warmup_epochs = 1  # warmup epochs
        self.batch_size = int(192 / n_gpus)  # batch size of each gpu

        # multi GPU and AMP
        self.num_workers = int(24 / n_gpus)  # the number of workers
        self.amp = True  # automatic mixed precision (AMP)

        # feedback
        self.feedback_iteration = 1000
        self.checkpoint_epoch = 20

        # log
        log_name = get_log_name(model_type, foundation_models)
        self.save_dir = Path("logs/" + log_name)

        # model configs
        self.model_type = model_type
        if checkpoint is not None:
            self.checkpoint = self.save_dir / "models" / checkpoint
        else:
            self.checkpoint = None
