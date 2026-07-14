import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_curve(loss_history, lr_history, config, current_epoch=None, current_loss=None,
                    accuracy_history=None, accuracy_epochs=None,
                    val_accuracy=None):
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    save_path_png    = os.path.join(config.MODEL_DIR, "loss_curve.png")
    save_path_lr_png = os.path.join(config.MODEL_DIR, "lr_curve.png")
    save_path_acc_png = os.path.join(config.MODEL_DIR, "accuracy_curve.png")
    save_path_txt    = os.path.join(config.MODEL_DIR, "training_log.txt")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'SigLIP2 Training Loss '
              f'(Epoch {current_epoch if current_epoch is not None else "?"})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path_png, dpi=150)
    plt.close()

    if lr_history:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(lr_history)), lr_history, linewidth=2, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(save_path_lr_png, dpi=150)
        plt.close()

    # 绘制准确率曲线
    if accuracy_history is not None and accuracy_epochs is not None and len(accuracy_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(accuracy_epochs, [acc * 100 for acc in accuracy_history],
                linewidth=2, color='green', marker='o', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'SigLIP2 Evaluation Accuracy '
                  f'(Best: {max(accuracy_history)*100:.1f}%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path_acc_png, dpi=150)
        plt.close()

    if current_epoch is not None and current_loss is not None:
        if not os.path.exists(save_path_txt):
            with open(save_path_txt, 'w', encoding='utf-8') as f:
                f.write("# SigLIP2 Training Log\n")
                f.write(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Config: Epochs={config.EPOCHS}, LR={config.LEARNING_RATE}, "
                        f"Scheduler={config.SCHEDULER_TYPE}, Model={config.SIGLIP_MODEL}\n")
                f.write("#" + "=" * 60 + "\n")
                f.write("# Format: epoch, avg_loss, min_loss, max_loss, learning_rate, val_accuracy, epoch_complete_time\n")

        with open(save_path_txt, 'a', encoding='utf-8') as f:
            min_loss   = min(loss_history)
            max_loss   = max(loss_history)
            current_lr = lr_history[-1] if lr_history else "N/A"
            complete_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            val_acc_str = f"{val_accuracy:.4f}" if val_accuracy is not None else ""

            if current_lr == "N/A":
                f.write(f"{current_epoch}, {current_loss:.6f}, "
                        f"{min_loss:.6f}, {max_loss:.6f}, N/A, {val_acc_str}, {complete_time}\n")
            else:
                f.write(f"{current_epoch}, {current_loss:.6f}, "
                        f"{min_loss:.6f}, {max_loss:.6f}, {current_lr:.6e}, {val_acc_str}, {complete_time}\n")


def save_training_summary(loss_history, lr_history, config, final_epoch,
                          accuracy_history=None, accuracy_epochs=None):
    timestamp     = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path_txt = os.path.join(config.MODEL_DIR, f"loss_log_complete_{timestamp}.txt")

    with open(save_path_txt, 'w', encoding='utf-8') as f:
        f.write("# Complete SigLIP2 Training Log\n")
        f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Model: {config.SIGLIP_MODEL}\n")
        f.write(f"# Total epochs: {final_epoch + 1}\n")
        f.write(f"# Total iterations: {len(loss_history)}\n")
        f.write(f"# Config: LR={config.LEARNING_RATE}, Scheduler={config.SCHEDULER_TYPE}\n")
        f.write("#" + "=" * 60 + "\n\n")
        f.write(f"Final loss: {loss_history[-1]:.6f}\n")
        f.write(f"Min loss:   {min(loss_history):.6f}\n")
        f.write(f"Max loss:   {max(loss_history):.6f}\n")
        f.write(f"Avg loss:   {np.mean(loss_history):.6f}\n")
        if accuracy_history and len(accuracy_history) > 0:
            f.write(f"\n# Evaluation Accuracy\n")
            f.write(f"Best accuracy: {max(accuracy_history):.2%}\n")
            f.write(f"Final accuracy: {accuracy_history[-1]:.2%}\n")
        f.write("\n# Format: iteration_index, loss_value\n")
        for i, loss in enumerate(loss_history):
            f.write(f"{i}, {loss:.6f}\n")
        if lr_history:
            f.write("\n# Learning Rate History\n")
            for ep, lr in enumerate(lr_history):
                f.write(f"{ep}, {lr:.6e}\n")
        if accuracy_history and accuracy_epochs:
            f.write("\n# Evaluation Accuracy History\n")
            f.write("# Format: epoch, accuracy\n")
            for ep, acc in zip(accuracy_epochs, accuracy_history):
                f.write(f"{ep}, {acc:.6f}\n")

    print(f"Complete training log saved to: {save_path_txt}")
