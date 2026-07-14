import os

import torch


class CheckpointManager:
    def __init__(self, model_dir, model_name, max_checkpoints=5):
        self.model_dir       = model_dir
        self.model_name      = model_name
        self.max_checkpoints = max_checkpoints
        self.saved_epochs    = []
        os.makedirs(model_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, loss, lr=None, scheduler=None,
             extra_state_dicts=None):
        checkpoint_path = os.path.join(
            self.model_dir, f"{self.model_name}_epoch_{epoch}.pt"
        )
        checkpoint_data = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 loss,
            'lr':                   lr,
        }

        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state_dicts is not None:
            checkpoint_data.update(extra_state_dicts)

        torch.save(checkpoint_data, checkpoint_path)
        self.saved_epochs.append(epoch)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        if self.max_checkpoints > 0 and len(self.saved_epochs) > self.max_checkpoints:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        while len(self.saved_epochs) > self.max_checkpoints:
            oldest   = self.saved_epochs.pop(0)
            old_path = os.path.join(self.model_dir, f"{self.model_name}_epoch_{oldest}.pt")
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"  ✓ Removed old checkpoint: epoch_{oldest}")

    def save_best_loss(self, model, optimizer, epoch, loss, lr=None, scheduler=None,
                       extra_state_dicts=None):
        best_path = os.path.join(self.model_dir, f"{self.model_name}_best_loss.pt")
        checkpoint_data = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 loss,
            'lr':                   lr,
        }

        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state_dicts is not None:
            checkpoint_data.update(extra_state_dicts)

        torch.save(checkpoint_data, best_path)
        print(f"  ★ Best loss model saved: {best_path} (loss: {loss:.4f})")

    def save_best_eval(self, model, optimizer, epoch, eval_accuracy, loss, lr=None,
                       scheduler=None, extra_state_dicts=None):
        best_path = os.path.join(self.model_dir, f"{self.model_name}_best_eval.pt")
        checkpoint_data = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_accuracy':        eval_accuracy,
            'loss':                 loss,
            'lr':                   lr,
        }

        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state_dicts is not None:
            checkpoint_data.update(extra_state_dicts)

        torch.save(checkpoint_data, best_path)
        print(f"  ★ Best eval model saved: {best_path} (accuracy: {eval_accuracy:.2%})")

    def get_saved_epochs(self):
        return self.saved_epochs.copy()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, metric, epoch):
        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            return False

        improved = (metric < self.best_score - self.min_delta) if self.mode == 'min' \
                   else (metric > self.best_score + self.min_delta)

        if improved:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter    = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def get_best_info(self):
        return self.best_score, self.best_epoch
