#!/usr/bin/env python3
"""Generate a compact base-vs-gate-attention comparison PDF."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np

from report_base_results import parse_summary, add_table_page

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


def load_model_result(model_dir):
    viz = model_dir / "classification_viz"
    candidates = []
    for path in viz.glob("results_epoch_*.json"):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        candidates.append(data)
    if not candidates:
        raise FileNotFoundError(f"no classification results in {viz}")
    return max(candidates, key=lambda data: data["metrics"]["accuracy"])


def load_tests(root, subdir, names):
    output = []
    for label, dirname in names:
        rows, overall = parse_summary(root / subdir / dirname / "summary_report_siglip2_multiview.txt")
        output.append((label, rows, overall))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/liuqian/Aqcy/train0630_result_0715"))
    parser.add_argument("--output", type=Path, default=Path("/home/liuqian/Aqcy/train0630_result_0715/base_gate_comparison.pdf"))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    base = load_model_result(root / "trainresult_gifbox_0630_base")
    gate = load_model_result(root / "trainresult_gifbox_0630_gateattention")
    learnable = load_model_result(Path("/home/liuqian/Aqcy/train0630_result_0716/trainresult_gifbox_0630_learnable_gate"))
    base_tests = load_tests(root, "test_base", [("base no EMA", "test_result-noema"),
                                                   ("base EMA .3", "test_result_0.3"),
                                                   ("base EMA .7", "test_result_0.7")])
    gate_tests = load_tests(root, "test_gate", [("gate no EMA", "testresult_gate_noema"),
                                                   ("gate EMA .7", "testresult_gate_ema0.7"),
                                                   ("gate center-only EMA .7", "test_gate_center_only_ema07")])
    learnable_root = Path("/home/liuqian/Aqcy/train0630_result_0716")
    learnable_tests = load_tests(
        learnable_root, "test_716_learnablehgateatten",
        [("learnable gate EMA .7", ".")],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(12, 7)); ax.axis("off")
        ax.text(.5, .86, "SigLIP2 Base vs Gate Attention 对比分析", ha="center", fontsize=20, fontweight="bold")
        ax.text(.08, .68, f"Base 最佳验证：{base['metrics']['accuracy']:.2%}（Epoch {base['epoch']}）", fontsize=14)
        ax.text(.08, .61, f"Gate 最佳验证：{gate['metrics']['accuracy']:.2%}（Epoch {gate['epoch']}）", fontsize=14)
        base_test = base_tests[2][2]["mean_accuracy"]
        gate_test = gate_tests[1][2]["mean_accuracy"]
        ax.text(.08, .51, f"Base 外部测试（EMA=.7）：{base_test:.2f}%", fontsize=14)
        ax.text(.08, .44, f"Gate 外部测试（EMA=.7）：{gate_test:.2f}%", fontsize=14)
        ax.text(.08, .37, f"Gate 相对 Base：{gate_test-base_test:+.2f} 个百分点", fontsize=14)
        ax.text(.08, .26, "注意：外部测试报告仅包含2个视频，结果适合作为初步对比。", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        labels = ["Base", "Gate"]
        val = [base["metrics"]["accuracy"] * 100, gate["metrics"]["accuracy"] * 100]
        test = [base_test, gate_test]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(2); width=.34
        ax.bar(x-width/2, val, width, label="最佳验证", color="#66c2a5")
        ax.bar(x+width/2, test, width, label="外部测试 EMA=.7", color="#3288bd")
        ax.set_xticks(x, labels); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(50, 101)
        ax.set_title("Base 与 Gate Attention 准确率对比"); ax.grid(axis="y", alpha=.3); ax.legend()
        for i, v in enumerate(val): ax.text(i-width/2, v+.5, f"{v:.2f}", ha="center")
        for i, v in enumerate(test): ax.text(i+width/2, v+.5, f"{v:.2f}", ha="center")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        rows = []
        for name, collection in (("Base", base_tests), ("Gate", gate_tests)):
            for label, _, overall in collection:
                rows.append([name, label, f"{overall['mean_accuracy']:.2f}%",
                             f"{overall['mean_stability']:.4f}",
                             f"{overall['mean_response']:.1f}%",
                             f"{overall['mean_hold']:.1f}%"])
        add_table_page(pdf, "所有外部测试设置对比",
                       ["模型", "推理设置", "平均准确率", "稳定性", "响应率", "保持率"], rows)

        for title, data in (("Base 最佳验证混淆矩阵", base), ("Gate 最佳验证混淆矩阵", gate)):
            cm = np.array(data["confusion_matrix"]["matrix"])
            classes = data["confusion_matrix"]["classes"]
            fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        learn_test = learnable_tests[0][2]
        rows = [
            ["learnable_gate", f"Epoch {learnable['epoch']}", f"{learnable['metrics']['accuracy']:.2%}",
             f"{learn_test['mean_accuracy']:.2f}%", f"{learn_test['mean_stability']:.4f}",
             f"{learn_test['mean_response']:.1f}%", f"{learn_test['mean_hold']:.1f}%"],
        ]
        add_table_page(pdf, "learnable_gate 版本补充结果",
                       ["版本", "最佳轮次", "验证准确率", "外部测试准确率", "稳定性", "响应率", "保持率"], rows)

        labels = ["Base EMA .7", "Gate EMA .7", "Learnable gate"]
        values = [base_test, gate_test, learn_test["mean_accuracy"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, values, color=["#3288bd", "#d53e4f", "#5e4fa2"])
        ax.set_ylim(50, 85); ax.set_ylabel("Accuracy (%)")
        ax.set_title("三种版本外部视频测试准确率")
        ax.grid(axis="y", alpha=.3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + .5, f"{value:.2f}%", ha="center")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6)); ax.axis("off")
        base_pc = base["metrics"]["per_class_accuracy"]
        gate_pc = gate["metrics"]["per_class_accuracy"]
        rows = [[cls, f"{base_pc[cls]:.2%}", f"{gate_pc[cls]:.2%}",
                 f"{gate_pc[cls]-base_pc[cls]:+.2%}"] for cls in base_pc]
        table = ax.table(cellText=rows, colLabels=["类别", "Base", "Gate", "Gate-Base"],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2)
        ax.set_title("最佳验证集各类别准确率", fontsize=16, fontweight="bold", pad=20)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6)); ax.axis("off")
        gate_diff = gate_test - base_test
        lines = [
            f"1. Gate 最佳验证准确率为 {gate['metrics']['accuracy']:.2%}，Base 为 {base['metrics']['accuracy']:.2%}。",
            f"2. 在当前两个外部视频上，Gate EMA=.7 为 {gate_test:.2f}%，Base EMA=.7 为 {base_test:.2f}%，Gate 相差 {gate_diff:+.2f} 个百分点。",
            "3. 当前结果显示 Gate 在验证集上略优，但外部测试不一定同步提升；说明外部视频分布与训练/验证数据存在差异。",
            "4. Base 的 EMA=.7 外部测试为三组设置中最高；Gate 的 center-only EMA=.7 反而较低，说明文本/gate 分支对当前外部视频有帮助但仍需更多数据验证。",
            "5. 报告建议同时展示验证集和外部视频测试，不要仅依据训练/验证准确率宣称 Gate 泛化更好。",
            f"6. 新增 learnable_gate 版本验证准确率为 {learnable['metrics']['accuracy']:.2%}，外部测试为 {learn_test['mean_accuracy']:.2f}%；其验证集更高，但外部测试仍低于 Base EMA=.7。",
        ]
        ax.text(.05, .88, "结论与实验建议", fontsize=18, fontweight="bold")
        ax.text(.05, .78, "\n\n".join(lines), fontsize=12, va="top", wrap=True)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"Comparison PDF: {output}")


if __name__ == "__main__":
    main()
