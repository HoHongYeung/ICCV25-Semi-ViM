import argparse
import copy
import os
from typing import Any, Dict, Iterable, List

import yaml


__all__ = ["create_parser", "parse_commandline_args", "AttrDict"]


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class AttrDict(dict):
    """Dictionary with attribute-style access and recursive conversion."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for key, value in data.items():
            self[key] = self._wrap(value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return AttrDict(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def clone(self):
        return AttrDict(copy.deepcopy(dict(self)))

    def to_dict(self):
        result = {}
        for key, value in self.items():
            if isinstance(value, AttrDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, AttrDict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


def _default_cfg() -> AttrDict:
    cfg = AttrDict(
        MODEL=AttrDict(
            arch="cnn13",
            name="semivim",
            drop_ratio=0.0,
            SSMixup=AttrDict(
                enabled=False,
                nu=1.0,
                num_layers=1,
            ),
        ),
        TRAIN=AttrDict(
            print_freq=20,
            save_freq=0,
            epochs=400,
            epochs_arm=0,
            epochs_finetune=0,
            epochs_semi=400,
            threshold=0.95,
            semi=AttrDict(
                mu=1.0,
                weak_aug=None,
                strong_aug=None,
            ),
            optimizer=AttrDict(
                type="sgd",
                lr=0.1,
                momentum=0.9,
                nesterov=False,
                weight_decay=1e-4,
                scheduler="cos",
                min_lr=1e-4,
                steps=None,
                gamma=None,
                rampup_length=30,
                rampdown_length=None,
            ),
            LYAPEMA=AttrDict(
                enabled=False,
                kappa=1.0,
                lambda_=0.0,
            ),
        ),
        DATA=AttrDict(
            dataset=None,
            num_labels=None,
            sup_batch_size=64,
            usp_batch_size=64,
            workers=4,
            data_twice=False,
            data_idxs=False,
            label_exclude=False,
        ),
        CHECKPOINT=AttrDict(
            dir="./checkpoints",
        ),
        AUG=AttrDict(
            mixup_alpha=None,
        ),
        RUN=AttrDict(
            seed=42,
            log_interval=1,
        ),
    )
    return cfg


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semi-supervised Training -- PyTorch")

    parser.add_argument("--config", type=str, help="path to YAML config")
    parser.add_argument(
        "--opts",
        default=None,
        nargs="+",
        help="List of KEY VALUE pairs to override config options, e.g. TRAIN.LYAPEMA.enabled True",
    )

    # Log and save
    parser.add_argument("--print-freq", default=None, type=int, help="display frequency")
    parser.add_argument("--save-freq", default=None, type=int, help="checkpoint frequency")
    parser.add_argument("--save-dir", default=None, type=str, help="checkpoint directory")

    # Data
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--num-labels", type=int, help="number of labeled samples")
    parser.add_argument("--sup-batch-size", type=int, help="batch size for supervised data")
    parser.add_argument("--usp-batch-size", type=int, help="batch size for unsupervised data")
    parser.add_argument("--workers", type=int, help="number of data loading workers")
    parser.add_argument("--data-twice", type=str2bool, help="use two data stream")
    parser.add_argument("--data-idxs", type=str2bool, help="enable sample indices")
    parser.add_argument("--label-exclude", type=str2bool, help="exclude labeled samples in unsupervised batch")

    # Architecture
    parser.add_argument("--arch", "-a", type=str, help="backbone architecture")
    parser.add_argument("--model", type=str, help="trainer model key")
    parser.add_argument("--drop-ratio", type=float, help="dropout ratio")

    # Optimization
    parser.add_argument("--epochs", type=int, help="total epochs for legacy single-stage training")
    parser.add_argument("--epochs-arm", type=int, help="epochs for ARM pretraining stage")
    parser.add_argument("--epochs-finetune", type=int, help="epochs for supervised fine-tuning stage")
    parser.add_argument("--epochs-semi", type=int, help="epochs for semi-supervised stage")
    parser.add_argument("--optim", type=str, choices=["sgd", "adam"], help="optimizer type")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--momentum", type=float, help="momentum")
    parser.add_argument("--nesterov", type=str2bool, help="use nesterov momentum")
    parser.add_argument("--weight-decay", type=float, help="weight decay")
    parser.add_argument("--lr-scheduler", type=str, choices=["cos", "multistep", "exp-warmup", "none"], help="scheduler type")
    parser.add_argument("--min-lr", type=float, help="minimum learning rate")
    parser.add_argument("--steps", type=int, nargs="+", help="decay steps for multistep scheduler")
    parser.add_argument("--gamma", type=float, help="decay factor for multistep scheduler")
    parser.add_argument("--rampup-length", type=int, help="length of ramp-up")
    parser.add_argument("--rampdown-length", type=int, help="length of ramp-down")

    # FixMatch & Semi-supervision
    parser.add_argument("--threshold", type=float, help="pseudo-label confidence threshold (tau)")
    parser.add_argument("--semi-mu", type=float, help="weight on unsupervised loss (mu)")

    # EMA
    parser.add_argument("--train-lyapema-enabled", type=str2bool, help="enable LyapEMA teacher update")
    parser.add_argument("--train-lyapema-kappa", type=float, help="LyapEMA kappa")
    parser.add_argument("--train-lyapema-lambda", type=float, help="Lyapunov regularization coefficient")

    # Mixup / SSMixup
    parser.add_argument("--mixup-alpha", type=float, help="mixup alpha")
    parser.add_argument("--model-ssmixup-enabled", type=str2bool, help="enable SSMixup module")
    parser.add_argument("--model-ssmixup-nu", type=float, help="SSMixup residual scaling nu")
    parser.add_argument("--model-ssmixup-num-layers", type=int, help="number of layers participating in SSMixup")

    # Misc
    parser.add_argument("--seed", type=int, help="random seed")

    return parser


def parse_commandline_args() -> AttrDict:
    parser = create_parser()
    args = parser.parse_args()

    cfg = _default_cfg()

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        cfg = _merge_dicts(cfg, AttrDict(loaded))

    arg_cfg = _cfg_from_args(args)
    cfg = _merge_dicts(cfg, arg_cfg)

    if args.opts:
        if len(args.opts) % 2 != 0:
            raise ValueError("Override options should be KEY VALUE pairs.")
        for key, value in zip(args.opts[::2], args.opts[1::2]):
            _set_by_path(cfg, key, _parse_value(value))

    _expose_flattened(cfg)
    return cfg


def _cfg_from_args(args: argparse.Namespace) -> AttrDict:
    cfg = AttrDict()

    def assign(path: str, value: Any):
        if value is None:
            return
        _set_by_path(cfg, path, value)

    assign("MODEL.arch", args.arch)
    assign("MODEL.name", args.model)
    assign("MODEL.drop_ratio", args.drop_ratio)

    assign("TRAIN.print_freq", args.print_freq)
    assign("TRAIN.save_freq", args.save_freq)
    assign("CHECKPOINT.dir", args.save_dir)

    assign("DATA.dataset", args.dataset)
    assign("DATA.num_labels", args.num_labels)
    assign("DATA.sup_batch_size", args.sup_batch_size)
    assign("DATA.usp_batch_size", args.usp_batch_size)
    assign("DATA.workers", args.workers)
    assign("DATA.data_twice", args.data_twice)
    assign("DATA.data_idxs", args.data_idxs)
    assign("DATA.label_exclude", args.label_exclude)

    assign("TRAIN.epochs", args.epochs)
    assign("TRAIN.epochs_arm", args.epochs_arm)
    assign("TRAIN.epochs_finetune", args.epochs_finetune)
    assign("TRAIN.epochs_semi", args.epochs_semi)

    assign("TRAIN.optimizer.type", args.optim)
    assign("TRAIN.optimizer.lr", args.lr)
    assign("TRAIN.optimizer.momentum", args.momentum)
    assign("TRAIN.optimizer.nesterov", args.nesterov)
    assign("TRAIN.optimizer.weight_decay", args.weight_decay)
    assign("TRAIN.optimizer.scheduler", args.lr_scheduler)
    assign("TRAIN.optimizer.min_lr", args.min_lr)
    assign("TRAIN.optimizer.steps", args.steps)
    assign("TRAIN.optimizer.gamma", args.gamma)
    assign("TRAIN.optimizer.rampup_length", args.rampup_length)
    assign("TRAIN.optimizer.rampdown_length", args.rampdown_length)

    assign("TRAIN.threshold", args.threshold)
    assign("TRAIN.semi.mu", args.semi_mu)

    assign("TRAIN.LYAPEMA.enabled", args.train_lyapema_enabled)
    assign("TRAIN.LYAPEMA.kappa", args.train_lyapema_kappa)
    assign("TRAIN.LYAPEMA.lambda_", args.train_lyapema_lambda)

    assign("AUG.mixup_alpha", args.mixup_alpha)
    assign("MODEL.SSMixup.enabled", args.model_ssmixup_enabled)
    assign("MODEL.SSMixup.nu", args.model_ssmixup_nu)
    assign("MODEL.SSMixup.num_layers", args.model_ssmixup_num_layers)

    assign("RUN.seed", args.seed)

    return cfg


def _set_by_path(cfg: AttrDict, key: str, value: Any) -> None:
    keys = key.split(".")
    node = cfg
    for part in keys[:-1]:
        if part not in node or not isinstance(node[part], AttrDict):
            node[part] = AttrDict()
        node = node[part]
    node[keys[-1]] = AttrDict._wrap(value)


def _merge_dicts(base: AttrDict, other: AttrDict) -> AttrDict:
    result = base.clone()
    for key, value in other.items():
        if key not in result:
            result[key] = AttrDict._wrap(value)
        else:
            if isinstance(value, AttrDict) and isinstance(result[key], AttrDict):
                result[key] = _merge_dicts(result[key], value)
            else:
                result[key] = AttrDict._wrap(value)
    return result


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _expose_flattened(cfg: AttrDict) -> None:
    """Expose frequently used keys for backward compatibility with legacy code."""
    cfg.arch = cfg.MODEL.arch
    cfg.model = cfg.MODEL.name
    cfg.drop_ratio = cfg.MODEL.drop_ratio

    cfg.dataset = cfg.DATA.dataset
    cfg.num_labels = cfg.DATA.num_labels
    cfg.sup_batch_size = cfg.DATA.sup_batch_size
    cfg.usp_batch_size = cfg.DATA.usp_batch_size
    cfg.workers = cfg.DATA.workers
    cfg.data_twice = cfg.DATA.data_twice
    cfg.data_idxs = cfg.DATA.data_idxs
    cfg.label_exclude = cfg.DATA.label_exclude

    cfg.print_freq = cfg.TRAIN.print_freq
    cfg.save_freq = cfg.TRAIN.save_freq
    cfg.save_dir = cfg.CHECKPOINT.dir
    cfg.epochs = cfg.TRAIN.epochs

    cfg.lr = cfg.TRAIN.optimizer.lr
    cfg.momentum = cfg.TRAIN.optimizer.momentum
    cfg.nesterov = cfg.TRAIN.optimizer.nesterov
    cfg.weight_decay = cfg.TRAIN.optimizer.weight_decay
    cfg.lr_scheduler = cfg.TRAIN.optimizer.scheduler
    cfg.min_lr = cfg.TRAIN.optimizer.min_lr
    cfg.steps = cfg.TRAIN.optimizer.steps
    cfg.gamma = cfg.TRAIN.optimizer.gamma
    cfg.optim = cfg.TRAIN.optimizer.type
    cfg.rampup_length = cfg.TRAIN.optimizer.rampup_length
    cfg.rampdown_length = cfg.TRAIN.optimizer.rampdown_length

    cfg.threshold = cfg.TRAIN.threshold
    cfg.usp_weight = cfg.TRAIN.semi.mu
    cfg.ema_decay = None  # legacy field kept for compatibility
    cfg.mixup_alpha = cfg.AUG.mixup_alpha

    cfg.train_lyapema_enabled = cfg.TRAIN.LYAPEMA.enabled
    cfg.train_lyapema_kappa = cfg.TRAIN.LYAPEMA.kappa
    cfg.train_lyapema_lambda = cfg.TRAIN.LYAPEMA.lambda_

    cfg.model_ssmixup_enabled = cfg.MODEL.SSMixup.enabled
    cfg.model_ssmixup_nu = cfg.MODEL.SSMixup.nu
    cfg.model_ssmixup_num_layers = cfg.MODEL.SSMixup.num_layers

    cfg.semi_epochs = cfg.TRAIN.epochs_semi
    cfg.arm_epochs = cfg.TRAIN.epochs_arm
    cfg.finetune_epochs = cfg.TRAIN.epochs_finetune
    cfg.seed = cfg.RUN.seed
