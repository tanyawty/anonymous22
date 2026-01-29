import torch

def mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return ((pred - true) ** 2).mean()

def mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return (pred - true).abs().mean()

def rmse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - true) ** 2).mean() + 1e-12)

def seq_to_pf_ma_gap(y_seq: torch.Tensor):
    """    y_seq: (B, N, H) future return sequence (predicted or true).
    Returns: pf/ma/gap, each shaped (B, N).

    This matches the logic in your original gp_mech_multitask_stgnn.py:
      pf  = sum over horizon
      ma  = mean over horizon
      gap = max over horizon - min over horizon
    """
    pf = y_seq.sum(dim=-1)
    ma = y_seq.mean(dim=-1)
    gap = y_seq.max(dim=-1).values - y_seq.min(dim=-1).values
    return pf, ma, gap
