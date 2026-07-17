#!/usr/bin/env python3
"""Build a PDF report from the historical base-model results."""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


def parse_summary(path):
    text = path.read_text(encoding="utf-8")
    rows = []
    row_re = re.compile(
        r"^\s*(\S+)\s+(\S+)\s+(\d+)\s+([\d.]+)%\s+(\d+)/(\d+)\s+"
        r"([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)"
    )
    for line in text.splitlines():
        match = row_re.match(line)
        if not match:
            continue
        group, video, frames, acc, correct, total, stability, response, hold, latency = match.groups()
        rows.append({
            "group": group, "video": video, "frames": int(frames),
            "accuracy": float(acc), "correct": int(correct), "total": int(total),
            "stability": float(stability), "response": float(response),
            "hold": float(hold), "inference_ms": float(latency),
        })
    overall = re.search(r"平均准确率:\s+([\d.]+)%.*", text)
    mean_stability = re.search(r"平均稳定性:\s+([\d.]+)", text)
    mean_response = re.search(r"平均响应率:\s+([\d.]+)%", text)
    mean_hold = re.search(r"平均保持率:\s+([\d.]+)%", text)
    return rows, {
        "mean_accuracy": float(overall.group(1)) if overall else None,
        "mean_stability": float(mean_stability.group(1)) if mean_stability else None,
        "mean_response": float(mean_response.group(1)) if mean_response else None,
        "mean_hold": float(mean_hold.group(1)) if mean_hold else None,
    }


def add_table_page(pdf, title, columns, rows, fontsize=8):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.7)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#b0b0b0")
    for column in range(len(columns)):
        cell = table.get_celld()[(0, column)]
        cell.set_facecolor("#d9eaf7")
        cell.set_text_props(weight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("/home/liuqian/Aqcy/train0630_result_0715"))
    parser.add_argument("--output", type=Path,
                        default=Path("/home/liuqian/Aqcy/train0630_result_0715/base_results_analysis.pdf"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    train_dir = root / "trainresult_gifbox_0630_base"
    viz_dir = train_dir / "classification_viz"

    with (train_dir / "best_checkpoint_summary.json").open(encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    with (train_dir / "overfitting_diagnostics.json").open(encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    best_epoch = int(checkpoint["best_eval_epoch"])
    with (viz_dir / f"results_epoch_{best_epoch}.json").open(encoding="utf-8") as handle:
        best_results = json.load(handle)

    tests = []
    for label, directory in (("no EMA", "test_result-noema"),
                             ("EMA beta=0.3", "test_result_0.3"),
                             ("EMA beta=0.7", "test_result_0.7")):
        rows, overall = parse_summary(root / "test_base" / directory / "summary_report_siglip2_multiview.txt")
        tests.append((label, rows, overall))

    output.parent.mkdir(parents=True, exist_ok=True)
    # CSV files are deliberately simple and open directly in Excel.
    with (output.with_name(output.stem + "_test_details.csv")).open("w", encoding="utf-8") as handle:
        handle.write("setting,video,frames,accuracy,correct,total,stability,response,hold,inference_ms\n")
        for label, rows, _ in tests:
            for row in rows:
                handle.write(",".join([label] + [str(row[k]) for k in (
                    "video", "frames", "accuracy", "correct", "total", "stability",
                    "response", "hold", "inference_ms")]) + "\n")

    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis("off")
        ax.text(0.5, 0.86, "SigLIP2 Multi-View Base 模型结果分析", ha="center", fontsize=21, fontweight="bold")
        ax.text(0.5, 0.77, "数据集：gift_m6_picture_train40_mult_0630", ha="center", fontsize=13)
        ax.text(0.08, 0.62, f"最佳验证轮次：Epoch {best_epoch}", fontsize=13)
        ax.text(0.08, 0.56, f"最佳验证准确率：{checkpoint['best_eval_accuracy']:.2%} ({best_results['metrics']['correct']}/{best_results['metrics']['total']})", fontsize=13)
        ax.text(0.08, 0.50, f"最佳 loss：{checkpoint['best_loss']:.4f}（Epoch {checkpoint['best_loss_epoch']}）", fontsize=13)
        ax.text(0.08, 0.44, f"最新 train-val accuracy gap：{checkpoint['latest_generalization_gap']['accuracy_gap']:.2%}", fontsize=13)
        ax.text(0.08, 0.34, "评价方式：SupCon 特征 + 类别中心余弦相似度分类", fontsize=12)
        ax.text(0.08, 0.29, "测试集：两个完整视频；EMA 只用于推理时平滑，不改变训练权重", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        epochs = [int(x["epoch"]) for x in diagnostics]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(epochs, [x["train_accuracy"] * 100 for x in diagnostics], "o-", label="train")
        axes[0].plot(epochs, [x["val_accuracy"] * 100 for x in diagnostics], "o-", label="val")
        axes[0].set(title="训练/验证准确率", xlabel="Epoch", ylabel="Accuracy (%)")
        axes[0].grid(alpha=.3); axes[0].legend()
        axes[1].plot(epochs, [x["accuracy_gap"] * 100 for x in diagnostics], "o-", color="#d95f02")
        axes[1].axhline(0, color="black", lw=.8)
        axes[1].set(title="Train - Val 准确率差距", xlabel="Epoch", ylabel="Gap (percentage points)")
        axes[1].grid(alpha=.3)
        fig.suptitle("训练过程与泛化情况", fontsize=15, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        per_class = best_results["metrics"]["per_class_accuracy"]
        cm = np.array(best_results["confusion_matrix"]["matrix"])
        rows = [[cls, f"{per_class[cls]:.2%}", str(int(cm[i].sum() - cm[i, i])),
                 ", ".join(f"{best_results['confusion_matrix']['classes'][j]}:{int(v)}" for j, v in enumerate(cm[i]) if v and j != i)]
                for i, cls in enumerate(best_results["confusion_matrix"]["classes"])]
        add_table_page(pdf, f"最佳验证结果（Epoch {best_epoch}）", ["类别", "准确率", "错误数", "主要错误"], rows)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(6), best_results["confusion_matrix"]["classes"])
        ax.set_yticks(range(6), best_results["confusion_matrix"]["classes"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("最佳验证混淆矩阵")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * .5 else "black")
        fig.colorbar(im, ax=ax); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        summary_rows = []
        for label, rows, overall in tests:
            summary_rows.append([label, f"{overall['mean_accuracy']:.2f}%",
                                 f"{overall['mean_stability']:.4f}",
                                 f"{overall['mean_response']:.1f}%",
                                 f"{overall['mean_hold']:.1f}%"])
        add_table_page(pdf, "base 推理阶段 EMA 对比", ["设置", "平均准确率", "平均稳定性", "响应率", "保持率"], summary_rows)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        labels = [x[0] for x in tests]
        accs = [x[2]["mean_accuracy"] for x in tests]
        stabs = [x[2]["mean_stability"] for x in tests]
        axes[0].bar(labels, accs, color=["#999999", "#66c2a5", "#3288bd"])
        axes[0].set_title("平均视频准确率"); axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(70, 82)
        axes[1].bar(labels, stabs, color=["#999999", "#66c2a5", "#3288bd"])
        axes[1].set_title("平均稳定性"); axes[1].set_ylabel("Stability"); axes[1].set_ylim(0, 1)
        for ax in axes: ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.3)
        fig.suptitle("EMA 对 base 推理结果的影响", fontsize=15, fontweight="bold")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6)); ax.axis("off")
        best_test = max(tests, key=lambda item: item[2]["mean_accuracy"])
        conclusions = [
            f"1. 最佳验证准确率为 {checkpoint['best_eval_accuracy']:.2%}，出现在 Epoch {best_epoch}。",
            f"2. 验证集最主要的错误类别为 M5，准确率为 {per_class['M5']:.2%}；混淆矩阵显示 M5 主要被判为 M4。",
            f"3. base 外部视频测试平均准确率从 no EMA 的 {tests[0][2]['mean_accuracy']:.2f}% 提升到 beta=0.7 的 {tests[2][2]['mean_accuracy']:.2f}%，提升 {tests[2][2]['mean_accuracy']-tests[0][2]['mean_accuracy']:.2f} 个百分点。",
            f"4. beta=0.7 的平均稳定性为 {tests[2][2]['mean_stability']:.4f}，高于 no EMA 的 {tests[0][2]['mean_stability']:.4f}。",
            "5. 因此可将 base_best_eval 权重作为模型权重，并在推理阶段启用 EMA；报告中应明确 EMA 是推理后处理。",
            "6. 外部测试目前只有2个视频，结论应表述为初步结果，后续应增加独立连续操作视频。",
        ]
        ax.text(.05, .88, "结论与报告建议", fontsize=18, fontweight="bold")
        ax.text(.05, .78, "\n\n".join(conclusions), fontsize=12, va="top", wrap=True)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"PDF report: {output}")
    print(f"CSV details: {output.with_name(output.stem + '_test_details.csv')}")


if __name__ == "__main__":
    main()
