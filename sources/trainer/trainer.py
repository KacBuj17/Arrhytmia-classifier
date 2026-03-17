import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils.project_paths import from_root

EPOCHS = 50
best_model_save_dir = from_root("resources/checkpoints")
def create_trainer():
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode="min"
    )

    lr_logger = LearningRateMonitor()

    logger = TensorBoardLogger(save_dir="logs", name="ECG_Classification")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{best_model_save_dir}",
        filename="best-lstm-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        check_val_every_n_epoch=1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        devices=1 if torch.cuda.is_available() else None,
        logger=logger,
    )

    return trainer