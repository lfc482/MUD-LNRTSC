from typing import Dict

import torch
import torch.nn.functional as F

from .config import Config


def smooth_one_hot(y: torch.Tensor, num_classes: int, smoothing: float):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y_smooth = torch.full((y.size(0), num_classes), off_value, device=y.device)
    y_smooth.scatter_(1, y.unsqueeze(1), on_value)
    return y_smooth


def kl_soft_loss(logits: torch.Tensor, target_prob: torch.Tensor, reduction="none"):
    log_prob = F.log_softmax(logits, dim=1)
    loss = F.kl_div(log_prob, target_prob, reduction="none").sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    return loss


def build_expert_soft_target(r_mean: torch.Tensor):
    p1 = torch.clamp(r_mean, 0.0, 1.0)
    p0 = 1.0 - p1
    return torch.stack([p0, p1], dim=1)


def compute_uncertainties(batch: Dict, x, y, logits, recon, cfg: Config):
    """Compute practical multi-source uncertainty components."""
    prob = F.softmax(logits.detach(), dim=1)
    confidence = prob.max(dim=1).values
    risk_window = batch["risk_window"].to(x.device)

    r_max = risk_window.max(dim=1).values
    r_mean = risk_window.mean(dim=1)
    high_ratio = (risk_window >= cfg.risk_threshold).float().mean(dim=1)

    # Expert criticality: high when maximum risk probability is close to the threshold.
    u_exp = torch.exp(-torch.abs(r_max - cfg.risk_threshold) / cfg.expert_sigma)
    u_exp = torch.clamp(u_exp, 0.0, 1.0)

    # Window-level weak-label expansion uncertainty.
    u_win_pos = 1.0 - high_ratio
    u_win_neg = torch.clamp(1.0 - torch.abs(r_max - cfg.risk_threshold) / cfg.risk_threshold, 0.0, 1.0)
    u_win = torch.where(y == 1, u_win_pos, u_win_neg)
    u_win = torch.clamp(u_win, 0.0, 1.0)

    # Boundary transition uncertainty: risk fluctuation + threshold criticality proxy.
    risk_var = torch.var(risk_window, dim=1)
    if torch.max(risk_var) > torch.min(risk_var):
        risk_var_norm = (risk_var - risk_var.min()) / (risk_var.max() - risk_var.min() + 1e-8)
    else:
        risk_var_norm = torch.zeros_like(risk_var)
    u_boundary = cfg.boundary_beta * risk_var_norm + (1 - cfg.boundary_beta) * u_exp
    u_boundary = torch.clamp(u_boundary, 0.0, 1.0)

    # Cross-well distribution-shift proxy: reconstruction error normalized within batch.
    # In deployment, this can be combined with feature-space Mahalanobis distance.
    recon_err = ((recon.detach() - x) ** 2).mean(dim=(1, 2))
    if torch.max(recon_err) > torch.min(recon_err):
        u_shift = (recon_err - recon_err.min()) / (recon_err.max() - recon_err.min() + 1e-8)
    else:
        u_shift = torch.zeros_like(recon_err)
    u_shift = torch.clamp(u_shift, 0.0, 1.0)

    return {
        "confidence": confidence,
        "r_mean": r_mean,
        "u_exp": u_exp,
        "u_win": u_win,
        "u_boundary": u_boundary,
        "u_shift": u_shift,
    }


def source_aware_loss(batch, x, y, student_logits, teacher_logits, recon, epoch: int, cfg: Config):
    """Source-aware soft supervision with auxiliary reconstruction loss."""
    y_smooth = smooth_one_hot(y, cfg.num_classes, cfg.label_smoothing)

    if epoch <= cfg.warmup_epochs:
        cls_loss_vec = kl_soft_loss(student_logits, y_smooth, reduction="none")
        recon_loss = F.mse_loss(recon, x)
        total_loss = cls_loss_vec.mean() + cfg.recon_lambda * recon_loss
        return total_loss, {
            "cls_loss": cls_loss_vec.mean().item(),
            "recon_loss": recon_loss.item(),
            "mean_conf": F.softmax(student_logits.detach(), dim=1).max(dim=1).values.mean().item(),
        }

    with torch.no_grad():
        teacher_prob = F.softmax(teacher_logits, dim=1)
        teacher_conf = teacher_prob.max(dim=1).values

    unc = compute_uncertainties(batch, x, y, student_logits, recon, cfg)
    confidence = unc["confidence"]
    u_exp = unc["u_exp"]
    u_win = unc["u_win"]
    u_boundary = unc["u_boundary"]
    u_shift = unc["u_shift"]
    expert_target = build_expert_soft_target(unc["r_mean"])

    loss_raw = kl_soft_loss(student_logits, y_smooth, reduction="none")
    loss_expert = kl_soft_loss(student_logits, expert_target, reduction="none")
    loss_teacher = kl_soft_loss(student_logits, teacher_prob.detach(), reduction="none")

    w_raw = (1.0 - u_exp) * (1.0 - u_win) * (1.0 - u_boundary)
    w_expert = u_exp + u_win + u_boundary
    w_teacher = teacher_conf * (1.0 - cfg.teacher_suppress_shift * u_shift) * (1.0 - cfg.teacher_suppress_boundary * u_boundary)
    w_teacher = torch.clamp(w_teacher, min=0.0)

    w_sum = w_raw + w_expert + w_teacher + 1e-8
    w_raw = w_raw / w_sum
    w_expert = w_expert / w_sum
    w_teacher = w_teacher / w_sum

    cls_loss_vec = w_raw * loss_raw + w_expert * loss_expert + w_teacher * loss_teacher

    sample_weight = 1.0 + confidence - u_exp - u_win + 0.3 * u_boundary
    sample_weight = torch.clamp(sample_weight, cfg.sample_weight_min, cfg.sample_weight_max)
    sample_weight = sample_weight / (sample_weight.mean() + 1e-8)

    cls_loss = (sample_weight * cls_loss_vec).mean()
    recon_loss = F.mse_loss(recon, x)
    total_loss = cls_loss + cfg.recon_lambda * recon_loss

    return total_loss, {
        "cls_loss": cls_loss.item(),
        "recon_loss": recon_loss.item(),
        "mean_conf": confidence.mean().item(),
        "u_exp": u_exp.mean().item(),
        "u_win": u_win.mean().item(),
        "u_boundary": u_boundary.mean().item(),
        "u_shift": u_shift.mean().item(),
    }
