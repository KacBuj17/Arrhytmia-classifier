from pytorch_lightning.tuner.tuning import Tuner


def tune_lr(trainer, model, train_dataloader, validation_dataloader):
    res = Tuner(trainer).lr_find(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
        early_stop_threshold=1000.0,
        max_lr=0.3
    )

    print(f"Suggested LR: {res.suggestion()}")
    model.lr = res.suggestion()
