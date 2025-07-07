import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

from data.loaders import Loader
from models.regression import GafToProbRegressor  # ★ NEW model

import warnings
warnings.simplefilter("ignore")


class Opt:
    """옵션 하이퍼파라미터를 한곳에 모아둔 클래스"""
    def __init__(self):
        # ─── 데이터 ───────────────────────────────────
        self.n_timestamps = 18              # 캔들(타임스텝) 수
        self.test_size = 0.15
        self.val_size = 0.15
        self.seed = 0
        self.x_transformation = "gasf"     # 입력 변환 방법
        self.y_transformation = "binary"   # ★ 각 캔들 0/1 레이블이라고 가정

        # ─── 모델 ───────────────────────────────────
        self.encode_channels   = [12, 24]
        self.encode_block_type = "DenseNet"
        self.encode_block_dim  = 3

        # ─── 학습 ───────────────────────────────────
        self.train_batch_size = 48
        self.val_batch_size   = 48
        self.test_batch_size  = 1
        self.learning_rate    = 1e-3
        self.weight_decay     = 1e-5
        self.num_epochs       = 70
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


opt = Opt()

# ─────────────────────────────────────────────────────
# 1) 데이터 로드
# ─────────────────────────────────────────────────────
loader = Loader(
    n_timestamps=opt.n_timestamps,
    test_size=opt.test_size,
    val_size=opt.val_size,
    seed=opt.seed,
    x_transformation=opt.x_transformation,
    y_transformation=opt.y_transformation,  # ★ binary 레이블
)

train_set, val_set, test_set = loader.air_quality()

train_dataloader = DataLoader(train_set, batch_size=opt.train_batch_size, shuffle=True)
val_dataloader   = DataLoader(val_set,   batch_size=opt.val_batch_size)
test_dataloader  = DataLoader(test_set,  batch_size=opt.test_batch_size)

# ─────────────────────────────────────────────────────
# 2) 모델 정의
# ─────────────────────────────────────────────────────
# 샘플 하나로 입력/출력 크기 추정
_, x_sample, _, y_sample = train_set[0]

model = GafToProbRegressor(
    in_channels=x_sample.shape[0],
    n_timestamps=opt.n_timestamps,
    encode_channels=opt.encode_channels,
    encode_block_type=opt.encode_block_type,
    encode_block_dim=opt.encode_block_dim,
    image_size=x_sample.shape[1],
).to(opt.device)

summary(model, input_size=x_sample.shape)

# ─────────────────────────────────────────────────────
# 3) 손실·옵티마이저
# ─────────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss().to(opt.device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
)

training_losses, validation_losses = [], []

# ─────────────────────────────────────────────────────
# 4) 학습 루프
# ─────────────────────────────────────────────────────
for epoch in range(opt.num_epochs):
    # ── Train ───────────────────────────────────────
    model.train()
    epoch_losses = []
    for _, x, _, y in train_dataloader:
        x = x.to(opt.device)
        y = y.to(opt.device).float()   # (B, n_timestamps) — 0/1

        logits = model(x)             # (B, n_timestamps)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.detach())

    mean_train_loss = torch.stack(epoch_losses).mean()
    training_losses.append(mean_train_loss.cpu().item())

    # ── Validation ──────────────────────────────────
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _, x_v, _, y_v in val_dataloader:
            x_v = x_v.to(opt.device)
            y_v = y_v.to(opt.device).float()
            logits_v = model(x_v)
            loss_v = criterion(logits_v, y_v)
            val_losses.append(loss_v.detach())

    mean_val_loss = torch.stack(val_losses).mean()
    validation_losses.append(mean_val_loss.cpu().item())

    print(f"Epoch [{epoch+1}/{opt.num_epochs}]  train={mean_train_loss:.4f}  val={mean_val_loss:.4f}")

# ─────────────────────────────────────────────────────
# 5) 학습/검증 loss 그래프
# ─────────────────────────────────────────────────────
plt.plot(np.arange(opt.num_epochs), training_losses, "ro", label="training")
plt.plot(np.arange(opt.num_epochs), validation_losses, "b", label="validation")
plt.xlabel("Epoch")
plt.ylabel("BCE loss")
plt.title("Training / Validation Loss")
plt.legend()
plt.show()

# ─────────────────────────────────────────────────────
# 6) 테스트
# ─────────────────────────────────────────────────────
model.eval()
all_test_losses = []
with torch.no_grad():
    for i, (_, x_t, _, y_t) in enumerate(test_dataloader):
        x_t = x_t.to(opt.device)
        y_t = y_t.to(opt.device).float()
        logits_t = model(x_t)
        loss_t = criterion(logits_t, y_t)
        all_test_losses.append(loss_t.cpu().item())

        if i % 50 == 0:
            probs = torch.sigmoid(logits_t)[0].cpu().numpy()   # (n_timestamps,)
            print(f"Sample {i}: probs = {np.round(probs,3)} …")

print("\nTest results:")
print("mean  BCE:", np.mean(all_test_losses))
print("std   BCE:", np.std(all_test_losses))
print("min   BCE:", np.min(all_test_losses))
print("max   BCE:", np.max(all_test_losses))
print("median BCE:", np.median(all_test_losses))