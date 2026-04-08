import torch
import pickle

from dataset.dataset import prepare_dataloaders
from model.model import LSTMClassifier
from trainer.trainer import create_trainer
from tuner.tuner import tune_lr
from utils.project_paths import from_root

ENCODER_PATH = from_root("resources/checkpoints/label_encoder.pkl")


def main():
    torch.set_float32_matmul_precision('medium')

    train_loader, val_loader, test_loader, label_encoder, class_weights = prepare_dataloaders()

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    model = LSTMClassifier(
        input_size=1,
        hidden_size=128,
        num_classes=len(label_encoder.classes_),
        class_weights=class_weights
    )

    trainer = create_trainer()
    tune_lr(trainer, model, train_loader, val_loader)

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
