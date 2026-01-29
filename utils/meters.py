from dataclasses import dataclass

@dataclass
class AverageMeter:
    """Track and average scalar values (e.g., losses) over steps."""
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, v: float, n: int = 1) -> None:
        self.val = float(v)
        self.sum += float(v) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)
