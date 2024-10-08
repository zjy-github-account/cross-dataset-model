from einops import rearrange
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torch.nn import functional as F

class _PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 22, embedding_size: int = 40):
        super(_PatchEmbedding, self).__init__()
        self.embedding_size = embedding_size

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (in_channels, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5)
        )

        self.projection = nn.Conv2d(40, self.embedding_size, (1, 1))

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        x = self.shallownet(x)
        x = self.projection(x)
        x = torch.permute(torch.squeeze(x), (0, 2, 1))
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling = self.emb_size ** (1 / 2)
        att = torch.nn.functional.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(_ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, embedding_size: int, expansion: int, dropout: float):
        self.embedding_size = embedding_size
        self.expansion = expansion
        self.dropout = dropout
        super(_FeedForwardBlock, self).__init__(
            nn.Linear(self.embedding_size, self.expansion * self.embedding_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.expansion * self.embedding_size, self.embedding_size)
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embedding_size: int, num_heads: int = 10, dropout: float = 0.5,
                 forward_expansion: int = 4, forward_dropout: float = 0.5):
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.forward_dropout = forward_dropout
        super(_TransformerEncoderBlock, self).__init__(
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(self.embedding_size),
                _MultiHeadAttention(self.embedding_size, self.num_heads, self.dropout),
                nn.Dropout(self.dropout)
            )),
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(self.embedding_size),
                _FeedForwardBlock(self.embedding_size, expansion=self.forward_expansion,
                                  dropout=self.forward_dropout),
                nn.Dropout(self.dropout)
            ))
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, embedding_size: int):
        self.depth = depth
        self.embedding_size = embedding_size
        super(_TransformerEncoder, self).__init__(
            *[_TransformerEncoderBlock(self.embedding_size) for _ in range(self.depth)])


class _ClassificationHead(nn.Module):
    def __init__(self, embedding_size: int, n_classes: int, input_size_cls: int = 2440):
        super(_ClassificationHead, self).__init__()
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.head = nn.Sequential(
            nn.Linear(input_size_cls, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.head(x)
        return x, out


class EEGConformerModule(nn.Module):
    def __init__(self, in_channels: int = 22, embedding_size: int = 40, depth: int = 6,
                 n_classes: int = 4, input_size_cls: int = 2440):
        super(EEGConformerModule, self).__init__()
        self.embedding_size = embedding_size
        self.depth = depth
        self.n_classes = n_classes
        self.patch_embedding = _PatchEmbedding(in_channels, self.embedding_size)
        self.transformer_encoder = _TransformerEncoder(self.depth, self.embedding_size)
        self.classification_head = _ClassificationHead(self.embedding_size,
                                                       self.n_classes,
                                                       input_size_cls=input_size_cls)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        _, x = self.classification_head(x)
        return x


# class EEGConformer(ClassificationModule):
    # def __init__(self, in_channels: int = 22, embedding_size: int = 40, depth: int = 6,
    #              n_classes: int = 4, input_size_cls: int = 2440, **kwargs):
    #     model = EEGConformerModule(in_channels, embedding_size, depth, n_classes,
    #                                input_size_cls)
    #     super(EEGConformer, self).__init__(model, n_classes, **kwargs)

# class ClassificationModule(pl.LightningModule):
#     def __init__(
#             self,
#             model,
#             n_classes,
#             lr=0.001,
#             weight_decay=0.0,
#             optimizer="adam",
#             scheduler=False,
#             max_epochs=1000,
#             warmup_epochs=20,
#             **kwargs
#     ):
#         super(ClassificationModule, self).__init__()
#         self.save_hyperparameters(ignore=["model"])
#         self.model = model
#
#     def forward(self, x):
#         return self.model(x)
#
#     def configure_optimizers(self):
#         betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
#         if self.hparams.optimizer == "adam":
#             optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
#                                          betas=betas,
#                                          weight_decay=self.hparams.weight_decay)
#         elif self.hparams.optimizer == "adamW":
#             optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
#                                           betas=betas,
#                                           weight_decay=self.hparams.weight_decay)
#         elif self.hparams.optimizer == "sgd":
#             optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
#                                         weight_decay=self.hparams.weight_decay)
#         else:
#             raise NotImplementedError
#         if self.hparams.scheduler:
#             scheduler = LambdaLR(optimizer,
#                                  linear_warmup_cosine_decay(self.hparams.warmup_epochs,
#                                                             self.hparams.max_epochs))
#             return [optimizer], [scheduler]
#         else:
#             return [optimizer]
#
#     def training_step(self, batch, batch_idx):
#         loss, acc = self.shared_step(batch, batch_idx, mode="train")
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss, acc = self.shared_step(batch, batch_idx, mode="val")
#         return {"val_loss": loss, "val_acc": acc}
#
#     def test_step(self, batch, batch_idx):
#         loss, acc = self.shared_step(batch, batch_idx, mode="test")
#         return {"test_loss": loss, "test_acc": acc}
#
#     def shared_step(self, batch, batch_idx, mode: str = "train"):
#         if mode == "train" and self.hparams.get("interaug", False):
#             x, y = interaug(batch)
#         else:
#             x, y = batch
#         y_hat = self.forward(x)
#         loss = F.cross_entropy(y_hat, y)
#         acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
#         self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
#         self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
#         return loss, acc
#
#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         x, _ = batch
#         return torch.argmax(self.forward(x), dim=-1)

def linear_warmup_cosine_decay(warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(
            1, total_steps - warmup_steps))

        # cosine decay
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return fn

def interaug(batch):
    x, y = batch
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    current = 0
    n_chunks = 8 if new_samples.shape[-1] % 8 == 0 else 7  # special case for BCIC III
    for cls in torch.unique(y):
        x_cls = x[y == cls]
        chunks = torch.cat(torch.chunk(x_cls, chunks=n_chunks, dim=-1))
        indices = np.random.choice(len(x_cls), size=(len(x_cls), n_chunks),
                                   replace=True)
        for idx in indices:
            # add offset
            idx += np.arange(0, chunks.shape[0], len(x_cls))

            # create new sample
            new_sample = chunks[idx]
            new_sample = new_sample.permute(1, 0, 2).reshape(
                1, x_cls.shape[1], x_cls.shape[2])
            new_samples[current] = new_sample
            new_labels[current] = cls
            current += 1

    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)

    # shuffle
    perm = torch.randperm(len(combined_x))
    combined_x = combined_x[perm]
    combined_y = combined_y[perm]

    return combined_x, combined_y
