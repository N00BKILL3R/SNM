from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def save_history_plots(history: List[Dict], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(Path(out_dir) / "history.csv", index=False)

    for key in ["train_loss", "val_loss", "val_acc", "val_psnr"]:
        if key not in df.columns:
            continue
        plt.figure()
        plt.plot(df["epoch"], df[key], marker="o")
        plt.title(key)
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.grid(True)
        plt.savefig(Path(out_dir) / f"{key}.png", bbox_inches="tight")
        plt.close()
