import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassF1Score


class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.4)
        self.lr = lr

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        )

    def forward(self, x):
        out, _ = self.lstm(x)

        # 🔥 zamiast last timestep → pooling
        out = torch.mean(out, dim=1)

        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        acc = (preds == y).float().mean()
        f1 = self.f1(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == y).float().mean()
        f1 = self.f1(preds, y)

        self.log('test_acc', acc)
        self.log('test_f1', f1)

        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
