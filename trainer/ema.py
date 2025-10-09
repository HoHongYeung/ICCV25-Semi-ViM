from typing import Dict, Iterable, Optional

import torch


class LyapEMA:
    """Lyapunov-stability-aware EMA teacher updater."""

    def __init__(
        self,
        kappa: float,
        lambda_: float,
        rollback_factor: float = 0.5,
    ) -> None:
        self.kappa = kappa
        self.lambda_ = lambda_
        self.rollback_factor = rollback_factor
        self.prev_V: Optional[float] = None

    @staticmethod
    def _vector_distance(student_params: Iterable[torch.nn.Parameter],
                         teacher_params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        students = list(student_params)
        teachers = list(teacher_params)
        device = students[0].device
        total = torch.zeros(1, device=device)
        for s_param, t_param in zip(students, teachers):
            total = total + torch.sum((s_param.data - t_param.data) ** 2)
        return 0.5 * total

    def regularization(self, student_model: torch.nn.Module,
                       teacher_model: torch.nn.Module) -> torch.Tensor:
        if self.lambda_ <= 0.0:
            param = next(student_model.parameters())
            return torch.zeros(1, device=param.device)
        student_params = [p for p in student_model.parameters()]
        teacher_params = [p for p in teacher_model.parameters()]
        reg = torch.zeros(1, device=student_params[0].device)
        for s_param, t_param in zip(student_params, teacher_params):
            reg = reg + torch.sum((s_param - t_param.detach()) ** 2)
        return self.lambda_ * reg

    def step(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            student_params = [p for p in student_model.parameters()]
            teacher_params = [p for p in teacher_model.parameters()]
            teacher_prev = [p.data.clone() for p in teacher_params]
            V_t = self._vector_distance(student_params, teacher_params)
            V_t_value = V_t.item()

            if self.prev_V is None:
                self.prev_V = V_t_value

            delta = V_t_value - self.prev_V
            alpha_tensor = torch.sigmoid(torch.tensor(-self.kappa * delta, device=V_t.device))
            alpha = alpha_tensor.item()

            one_minus_alpha = 1.0 - alpha
            for t_param, s_param in zip(teacher_params, student_params):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=one_minus_alpha)

            V_next = self._vector_distance(student_params, teacher_params)
            V_next_value = V_next.item()

            rolled_back = False
            if V_next_value >= V_t_value:
                rolled_back = True
                for t_param, prev in zip(teacher_params, teacher_prev):
                    t_param.data.copy_(prev)
                if optimizer is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = group["lr"] * self.rollback_factor
                self.kappa = self.kappa * (1.0 + (1.0 - self.rollback_factor))
            else:
                self.prev_V = V_next_value

        return {
            "alpha_t": alpha,
            "V_t": V_t_value,
            "V_next": V_next_value,
            "rolled_back": float(rolled_back),
        }
