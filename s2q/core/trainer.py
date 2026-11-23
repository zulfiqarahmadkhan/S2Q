"""Cellpose model training (Cellpose >= 4)."""
import logging
import math
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from cellpose import models, train
    CELLPOSE_AVAILABLE = True
except ImportError as e:
    CELLPOSE_AVAILABLE = False
    logger.warning(f"Cellpose not available for training: {e}")


class TrainingAborted(Exception):
    """Raised inside the training loop when the user presses Stop."""


class _EarlyStop(Exception):
    """Raised inside the training loop when the loss has stopped improving."""


class EarlyStopper:
    """Stops when the (smoothed) training loss has not improved by *min_delta* for *patience* epochs.

    The first epochs are ignored: Cellpose warms the learning rate up over 10 epochs.
    """

    def __init__(self, patience: int, min_delta: float = 0.005, warmup: int = 10, window: int = 5):
        self.patience, self.min_delta, self.warmup, self.window = patience, min_delta, warmup, window
        self.best = math.inf
        self.best_epoch = 0
        self.stopped_epoch = None

    def update(self, epoch: int, losses: list) -> bool:
        """*epoch* is 1-based; *losses* all epoch losses so far. True → stop now."""
        if not self.patience or epoch <= self.warmup:
            return False
        smooth = sum(losses[-self.window:]) / len(losses[-self.window:])
        if smooth < self.best * (1 - self.min_delta):
            self.best, self.best_epoch = smooth, epoch
        elif epoch - self.best_epoch >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False


class Trainer:
    """Fine-tune a Cellpose model on a folder of images + masks.

    Parameters
    ----------
    train_dir : str
        Folder containing the images and their mask files.
    mask_filter : str
        Suffix used by mask files, e.g. ``"_cp_masks"`` (``img.tif`` ↔ ``img_cp_masks.tif``).
    """

    def __init__(self, train_dir: str, mask_filter: str = "_cp_masks"):
        if not CELLPOSE_AVAILABLE:
            raise ImportError(
                "Cellpose is required for training. Install with: pip install cellpose[gui]"
            )
        from s2q.core.datasplit import find_pairs
        self.train_dir = train_dir
        self.mask_filter = mask_filter
        self.pairs = find_pairs(train_dir, mask_filter)
        if not self.pairs:
            raise ValueError(
                f"No image+mask pairs found in '{train_dir}'.\n"
                f"Make sure your mask files end with '{mask_filter}' "
                f"and images do not contain that suffix."
            )
        logger.info(f"Found {len(self.pairs)} image+mask pair(s)")

    def train(self, model_name: str, base_model: str = "cpsam",
              epochs: int = 100, learning_rate: float = 1e-5,
              weight_decay: float = 0.1, batch_size: int = 1,
              normalize: bool = True,
              save_path: str | None = None,
              progress_callback=None, should_stop=None,
              test_percent: float = 20, status_callback=None, early_stop_patience: int = 0) -> tuple:
        """Fine-tune *base_model*; return ``(model_path, train_losses, report)``.

        *test_percent* of the images (chosen at random, reproducibly) are kept
        out of training entirely. The split is saved next to the model
        (``<model>_s2q_split.json``) so "Test model" evaluates exactly those images.

        Parameters
        ----------
        base_model : str
            ``"cpsam"`` or a path to a previously fine-tuned model to continue from.
        learning_rate, weight_decay : float
            AdamW settings; Cellpose-SAM's recommended defaults are 1e-5 and 0.1.
        save_path : str, optional
            Directory where the model is written (Cellpose stores it in
            ``<save_path>/models/<model_name>``). Defaults to the training directory.
        progress_callback : callable, optional
            Called with ``(epoch, n_epochs, loss, pct)``: after every training step with
            ``loss = 0`` (progress only) and after every epoch with that epoch's mean loss.
        should_stop : callable, optional
            Polled before every training batch; returning True aborts training
            and raises :class:`TrainingAborted`.
        early_stop_patience : int
            Stop when the training loss has not improved for this many epochs (0 = off);
            the model trained so far is saved and ``report["early_stopped_at"]`` is set.
        """
        import torch
        from s2q.core.datasplit import save_split, split_pairs
        from s2q.core.device import use_gpu

        sp = Path(save_path) if save_path else Path(self.train_dir)
        sp.mkdir(parents=True, exist_ok=True)
        train_pairs, test_pairs = split_pairs(self.pairs, test_percent)
        if status_callback:
            status_callback(f"Training on {len(train_pairs)} image(s), {len(test_pairs)} kept for testing…")

        gpu = use_gpu()
        model = models.CellposeModel(gpu=gpu, pretrained_model=base_model)
        logger.info(
            f"Training from '{base_model}' on {'GPU' if gpu else 'CPU'}: {len(train_pairs)} train / "
            f"{len(test_pairs)} test images, epochs={epochs}, lr={learning_rate}, wd={weight_decay}, save_path={sp}"
        )

        # Progress after every step, the real loss after every epoch, and early stopping.
        # Cellpose itself only logs at epochs 1, 6, 11, 21, … which looked like a frozen bar.
        steps_per_epoch = max(1, math.ceil(_n_with_cells([m for _, m in train_pairs]) / batch_size))
        stopper = EarlyStopper(early_stop_patience)
        state = {"step": 0, "batch_losses": [], "epoch_losses": [], "last_emit": 0.0}
        original_get_batch, original_loss = train._get_batch, train._loss_fn_seg

        def _get_batch_hooked(*args, **kwargs):
            if should_stop and should_stop():
                raise TrainingAborted("Training stopped by user")
            return original_get_batch(*args, **kwargs)

        def _loss_hooked(*args, **kwargs):
            loss = original_loss(*args, **kwargs)
            state["batch_losses"].append(float(loss.detach()))
            state["step"] += 1
            step = state["step"]
            epoch_done = step % steps_per_epoch == 0
            epoch = step // steps_per_epoch if epoch_done else step // steps_per_epoch + 1
            pct = min(100, int(step / (steps_per_epoch * epochs) * 100))
            if epoch_done:
                ep_loss = sum(state["batch_losses"]) / len(state["batch_losses"])
                state["batch_losses"] = []
                state["epoch_losses"].append(ep_loss)
                if progress_callback:
                    progress_callback(epoch, epochs, ep_loss, pct)
                if stopper.update(epoch, state["epoch_losses"]):
                    raise _EarlyStop()
            elif progress_callback and time.time() - state["last_emit"] > 0.5:
                state["last_emit"] = time.time()
                progress_callback(epoch, epochs, 0.0, pct)
            return loss

        train._get_batch, train._loss_fn_seg = _get_batch_hooked, _loss_hooked
        try:
            # the test images are never shown to the training loop
            model_path, train_losses, _ = train.train_seg(
                model.net,
                train_files=[i for i, _ in train_pairs],
                train_labels_files=[m for _, m in train_pairs],
                load_files=True,
                normalize=normalize,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                n_epochs=epochs,
                # Cellpose skips images with < 5 cells by default (and crashes if none are left);
                # every labelled image the user provides should count
                min_train_masks=1,
                model_name=model_name,
                save_path=str(sp),
            )
        except _EarlyStop:
            # keep what has been learned: save the network as Cellpose would at the end
            model_path = sp / "models" / model_name
            model.net.save_model(str(model_path))
            train_losses = state["epoch_losses"]
            logger.info(f"Early stop at epoch {stopper.stopped_epoch}: no improvement for "
                        f"{early_stop_patience} epochs (best at epoch {stopper.best_epoch})")
        except torch.OutOfMemoryError as e:
            raise RuntimeError(
                "GPU ran out of memory during training. Close other GPU programs, "
                "switch to CPU in the sidebar, or use smaller / fewer images."
            ) from e
        finally:
            train._get_batch, train._loss_fn_seg = original_get_batch, original_loss

        if progress_callback and stopper.stopped_epoch is None:
            progress_callback(epochs, epochs, float(train_losses[-1]) if len(train_losses) else 0.0, 100)
        logger.info(f"Training complete — model saved at: {model_path}")
        split = save_split(str(model_path), "cellpose", train_pairs, test_pairs, base_model,
                           {"epochs": epochs, "learning_rate": learning_rate, "weight_decay": weight_decay,
                            "batch_size": batch_size, "normalize": normalize, "test_percent": test_percent,
                            "device": "GPU" if gpu else "CPU"},
                           train_losses)
        report = {"n_train": len(train_pairs), "n_test": len(test_pairs), "split_file": str(split)}
        if stopper.stopped_epoch is not None:
            report.update(early_stopped_at=stopper.stopped_epoch, best_epoch=stopper.best_epoch)
        return model_path, train_losses, report


def _n_with_cells(mask_files) -> int:
    """Images Cellpose actually trains on (it drops masks without any cell, min_train_masks=1)."""
    from skimage.io import imread
    n = 0
    for f in mask_files:
        try:
            n += int(imread(f).max() > 0)
        except Exception:
            n += 1
    return max(1, n)
