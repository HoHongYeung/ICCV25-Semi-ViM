#!coding:utf-8
import torch
from torch.nn import functional as F

import os
import datetime
from pathlib import Path
from collections import defaultdict
from itertools import cycle

from utils.ramps import exp_rampup
from utils.datasets import decode_label
from utils.data_utils import NO_LABEL
from trainer.ema import LyapEMA
from utils.SSMixup import build_ssmixup_inputs


class Trainer:

    def __init__(self, model, ema_model, optimizer, device, config):
        print("Semi-ViM")
        self.cfg = config
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.device = device

        train_cfg = config.TRAIN
        semi_cfg = train_cfg.semi
        lyap_cfg = train_cfg.LYAPEMA
        mix_cfg = config.MODEL.SSMixup

        self.lce_loss = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL)
        self.uce_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self.usp_weight = semi_cfg.mu
        self.threshold = train_cfg.threshold
        self.use_ssmixup = mix_cfg.enabled
        self.ssmixup_nu = mix_cfg.nu
        self.ssmixup_layers = mix_cfg.num_layers or 1
        self.ssmixup_eps = 1e-8

        self.lyapema = LyapEMA(
            kappa=lyap_cfg.kappa,
            lambda_=lyap_cfg.lambda_,
        ) if lyap_cfg.enabled else None

        self.rampup = exp_rampup(train_cfg.optimizer.rampup_length or 0)
        self.save_freq = train_cfg.save_freq
        self.print_freq = train_cfg.print_freq
        self.global_step = 0
        self.epoch = 0
        self.metric_history = []

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.save_dir = os.path.join(
            config.save_dir,
            f"{config.arch}-{config.model}_{config.dataset}-{config.num_labels}_{timestamp}",
        )

    def train_iteration(self, label_loader, unlab_loader, print_freq):
        loop_info = defaultdict(list)
        batch_idx, label_n, unlab_n = 0, 0, 0
        for ((x1, _), label_y), ((wu, su), unlab_y) in zip(cycle(label_loader), unlab_loader):
            self.global_step += 1
            batch_idx += 1
            label_x, weak_u, strong_u = x1.to(self.device), wu.to(self.device), su.to(self.device)
            label_y, unlab_y = label_y.to(self.device), unlab_y.to(self.device)
            self.decode_targets(unlab_y)
            lbs, ubs = x1.size(0), wu.size(0)

            sup_logits = self.model(label_x)
            supervised_loss = self.lce_loss(sup_logits, label_y)
            loss = supervised_loss
            loop_info["lloss"].append(supervised_loss.item())

            with torch.no_grad():
                teacher_weak = self.ema_model(weak_u)
                weak_probs = F.softmax(teacher_weak, dim=1)
                confidences, pseudo_labels = weak_probs.max(dim=1)

            mask = confidences.ge(self.threshold).float()
            strong_logits = self.model(strong_u)
            num_classes = strong_logits.size(1)
            uloss = self.uce_loss(strong_logits, pseudo_labels)
            if mask.sum() > 0:
                uloss = torch.sum(uloss * mask) / mask.sum()
            else:
                uloss = torch.tensor(0.0, device=self.device)
            loss = loss + (uloss * self.usp_weight)
            loop_info["uloss"].append(uloss.item())

            if self.use_ssmixup and ubs > 1:
                ssmix_loss, mix_metrics = self._compute_ssmixup(
                    strong_u, pseudo_labels, confidences, num_classes
                )
                loss = loss + ssmix_loss
                loop_info["umix"].append(mix_metrics.get("loss", 0.0))

            lyap_val = 0.0
            if self.lyapema is not None:
                lyap_reg = self.lyapema.regularization(self.model, self.ema_model)
                loss = loss + lyap_reg
                lyap_val = lyap_reg.item()
            loop_info["lyap"].append(lyap_val)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = self._grad_norm(self.model.parameters())
            loop_info["gnorm"].append(grad_norm)
            self.optimizer.step()

            ema_metrics = self._update_teacher()
            loop_info["alpha"].append(ema_metrics.get("alpha_t", 0.0))
            loop_info["vnorm"].append(ema_metrics.get("V_t", 0.0))
            if ema_metrics.get("rolled_back", 0.0):
                loop_info["rollback"].append(1.0)

            label_n += lbs
            unlab_n += ubs
            loop_info["lacc"].append(label_y.eq(sup_logits.max(1)[1]).float().sum().item())
            loop_info["uacc"].append(unlab_y.eq(strong_logits.max(1)[1]).float().sum().item())

            if print_freq > 0 and (batch_idx % print_freq) == 0:
                print(f"[train][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        self.update_bn(self.model, self.ema_model)
        print(f">>>[train]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    def _compute_ssmixup(self, inputs, pseudo_labels, confidences, num_classes):
        mix = build_ssmixup_inputs(
            inputs,
            pseudo_labels,
            confidences,
            threshold=self.threshold,
            eps=self.ssmixup_eps,
            device=self.device,
            num_classes=num_classes,
        )
        if hasattr(self.model, "forward_bidirectional"):
            y_fwd, y_bwd, h_fwd, h_bwd = self.model.forward_bidirectional(mix.inputs, self.ssmixup_layers)
        else:
            y_fwd = self.model(mix.inputs)
            y_bwd = self.ema_model(mix.inputs)
            h_fwd, h_bwd = y_fwd, y_bwd

        z = y_fwd - y_bwd
        gate = F.silu(z)
        h_mix = mix.psi.unsqueeze(1) * h_fwd + (1.0 - mix.psi).unsqueeze(1) * h_bwd
        logits = gate * y_fwd + gate * y_bwd + self.ssmixup_nu * h_mix

        mixed_targets = mix.targets
        log_prob = F.log_softmax(logits, dim=1)
        loss_vec = -torch.sum(mixed_targets * log_prob, dim=1)
        if mix.mask.sum() > 0:
            base_loss = torch.sum(loss_vec * mix.mask) / (mix.mask.sum() + self.ssmixup_eps)
        else:
            base_loss = torch.tensor(0.0, device=self.device)
        total_loss = self.usp_weight * base_loss
        metrics = {
            "loss": base_loss.item(),
            "psi": mix.psi.mean().item(),
        }
        return total_loss, metrics

    def _grad_norm(self, parameters):
        grads = [
            p.grad.detach().norm(2)
            for p in parameters
            if p.grad is not None
        ]
        if not grads:
            return 0.0
        stacked = torch.stack(grads)
        return torch.norm(stacked, 2).item()

    def _update_teacher(self):
        if self.lyapema is not None:
            return self.lyapema.step(self.model, self.ema_model, self.optimizer)
        with torch.no_grad():
            for t_param, s_param in zip(self.ema_model.parameters(), self.model.parameters()):
                t_param.data.copy_(s_param.data)
        return {"alpha_t": 1.0, "V_t": 0.0, "rolled_back": 0.0}

    def test_iteration(self, data_loader, print_freq):
        loop_info = defaultdict(list)
        label_n, unlab_n = 0, 0
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            lbs, ubs = data.size(0), -1

            outputs = self.model(data)
            ema_outputs = self.ema_model(data)

            label_n, unlab_n = label_n + lbs, unlab_n + ubs
            loop_info["lacc"].append(targets.eq(outputs.max(1)[1]).float().sum().item())
            loop_info["l2acc"].append(targets.eq(ema_outputs.max(1)[1]).float().sum().item())
            if print_freq > 0 and (batch_idx % print_freq) == 0:
                print(f"[test][{batch_idx:<3}]", self.gen_info(loop_info, lbs, ubs))
        print(f">>>[test]", self.gen_info(loop_info, label_n, unlab_n, False))
        return loop_info, label_n

    def train(self, label_loader, unlab_loader, print_freq=20):
        self.model.train()
        self.ema_model.train()
        with torch.enable_grad():
            return self.train_iteration(label_loader, unlab_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            return self.test_iteration(data_loader, print_freq)

    def arm_pretrain(self, unlab_loader, epochs, scheduler=None):
        self.model.train()
        self.ema_model.train()
        for ep in range(epochs):
            if scheduler is not None:
                scheduler.step()
            loop_info = defaultdict(list)
            for batch_idx, ((weak_u, strong_u), _) in enumerate(unlab_loader):
                weak_u = weak_u.to(self.device)
                strong_u = strong_u.to(self.device)

                self.optimizer.zero_grad()
                student_logits = self.model(strong_u)
                with torch.no_grad():
                    teacher_logits = self.ema_model(weak_u)

                loss = F.mse_loss(F.softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))
                lyap_val = 0.0
                if self.lyapema is not None:
                    lyap_reg = self.lyapema.regularization(self.model, self.ema_model)
                    loss = loss + lyap_reg
                    lyap_val = lyap_reg.item()
                loss.backward()
                grad_norm = self._grad_norm(self.model.parameters())
                self.optimizer.step()
                ema_metrics = self._update_teacher()

                loop_info["arm_loss"].append(loss.item())
                loop_info["gnorm"].append(grad_norm)
                loop_info["lyap"].append(lyap_val)
                loop_info["alpha"].append(ema_metrics.get("alpha_t", 0.0))
                loop_info["vnorm"].append(ema_metrics.get("V_t", 0.0))

                if self.print_freq > 0 and (batch_idx % self.print_freq) == 0:
                    print(f"[arm][{ep:03d}:{batch_idx:<3}]", self.gen_info(loop_info, 0, weak_u.size(0)))
            print(f">>>[arm][{ep:03d}]", self.gen_info(loop_info, 0, 1, False))

    def finetune_supervised(self, label_loader, test_loader, epochs, scheduler=None):
        self.model.train()
        self.ema_model.train()
        for ep in range(epochs):
            if scheduler is not None:
                scheduler.step()
            loop_info = defaultdict(list)
            for batch_idx, ((inputs, _), targets) in enumerate(label_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.lce_loss(logits, targets)
                lyap_val = 0.0
                if self.lyapema is not None:
                    lyap_reg = self.lyapema.regularization(self.model, self.ema_model)
                    loss = loss + lyap_reg
                    lyap_val = lyap_reg.item()

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = self._grad_norm(self.model.parameters())
                self.optimizer.step()
                ema_metrics = self._update_teacher()

                loop_info["lloss"].append(loss.item())
                loop_info["gnorm"].append(grad_norm)
                loop_info["lyap"].append(lyap_val)
                loop_info["alpha"].append(ema_metrics.get("alpha_t", 0.0))
                loop_info["vnorm"].append(ema_metrics.get("V_t", 0.0))
                loop_info["lacc"].append(targets.eq(logits.max(1)[1]).float().sum().item())

                if self.print_freq > 0 and (batch_idx % self.print_freq) == 0:
                    print(f"[finetune][{ep:03d}:{batch_idx:<3}]", self.gen_info(loop_info, inputs.size(0), 0))

            print(f">>>[finetune][{ep:03d}]", self.gen_info(loop_info, 1, 0, False))
            self.test(test_loader, self.print_freq)

    def loop(self, epochs, label_data, unlab_data, test_data, scheduler=None):
        best_acc, n, best_info = 0.0, 0.0, None
        for ep in range(epochs):
            self.epoch = ep
            if scheduler is not None:
                scheduler.step()
            print("------ Training epochs: {} ------".format(ep))
            self.train(label_data, unlab_data, self.print_freq)
            print("------ Testing epochs: {} ------".format(ep))
            info, n = self.test(test_data, self.print_freq)
            acc = sum(info["lacc"]) / n
            if acc > best_acc:
                best_acc, best_info = acc, info
            if self.save_freq != 0 and (ep + 1) % self.save_freq == 0:
                self.save(ep)
        print(f">>>[best]", self.gen_info(best_info, n, n, False))

    def update_bn(self, model, ema_model):
        for m2, m1 in zip(ema_model.named_modules(), model.named_modules()):
            if ("bn" in m2[0]) and ("bn" in m1[0]):
                bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
                bn2["running_mean"].data.copy_(bn1["running_mean"].data)
                bn2["running_var"].data.copy_(bn1["running_var"].data)
                bn2["num_batches_tracked"].data.copy_(bn1["num_batches_tracked"].data)

    def decode_targets(self, targets):
        label_mask = targets.ge(0)
        unlab_mask = targets.le(NO_LABEL)
        targets[unlab_mask] = decode_label(targets[unlab_mask])
        return label_mask, unlab_mask

    def gen_info(self, info, lbs, ubs, iteration=True):
        ret = []
        nums = {"l": lbs, "u": ubs, "a": lbs + ubs}
        for k, val in info.items():
            n = nums.get(k[0], nums["a"])
            v = val[-1] if iteration else sum(val)
            s = f"{k}: {v/n:.3%}" if k[-1] == "c" else f"{k}: {v:.5f}"
            ret.append(s)
        return "\t".join(ret)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {
                "epoch": epoch,
                "student": self.model.state_dict(),
                "teacher": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg.__dict__),
            }
            model_out_path.mkdir(parents=True, exist_ok=True)
            save_target = model_out_path / f"model_epoch_{epoch}.pth"
            torch.save(state, save_target)
            print(f"==> save model to {save_target}")
