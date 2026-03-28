from typing import Dict

import torch


class MetricsTop:
    @staticmethod
    def binary_acc_f1(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        preds = torch.argmax(logits, dim=-1)

        correct = (preds == labels).sum().item()
        total = labels.numel()
        acc = correct / max(total, 1)

        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        return {
            "acc": float(acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
