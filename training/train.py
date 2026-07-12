"""Train the 3-class subtitle classifier (no / N / S) and export to ONNX.

Reads dataset/no/, dataset/N/ and dataset/S/ (created with dump_frames.py +
manual sorting), trains a small CNN from scratch — no pretrained weights,
small enough for CPU — and exports subtitle_detector.onnx, which the app
runs via cv2.dnn. After export, the script sanity-checks that cv2.dnn
produces the same output as PyTorch.

Usage:
    pip install -r training/requirements.txt
    python training/train.py
    python training/train.py --dataset dataset --epochs 20 --out subtitle_detector.onnx
"""

import argparse
import os
import random
import sys

# Make the repo root importable when run as `python training/train.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch's ONNX exporter prints emoji, which crashes on non-UTF-8 Windows
# consoles (e.g. cp932). Force UTF-8 so training never dies on a print().
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
import torch.nn as nn

# Single source of truth for the contract shared with the app: class order
# (= dataset folder names = softmax output order) and input preprocessing.
from detector.presence import CLASS_NAMES, MODEL_INPUT_H, MODEL_INPUT_W

SEED = 42


class TinyPresenceNet(nn.Module):
    """4 conv blocks + global average pooling + one logit per class.

    Input:  (B, 3, MODEL_INPUT_H, MODEL_INPUT_W) RGB in [0, 1]
    Output: (B, len(CLASS_NAMES)) raw logits — softmax is appended at export.
    """

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16), block(16, 32), block(32, 64), block(64, 96)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, len(CLASS_NAMES)),
        )

    def forward(self, x):
        return self.head(self.features(x))


def load_folder(folder, class_index):
    """Load every image in `folder` resized to the model input. Returns
    (images NHWC float32 in [0,1] RGB, int64 labels)."""
    images, labels = [], []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        # cv2.imread can't open non-ASCII filenames on Windows (Japanese
        # video names); read the bytes with numpy and decode in memory.
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            print(f"  skipping unreadable file: {path}")
            continue
        img = cv2.resize(img, (MODEL_INPUT_W, MODEL_INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(img)
        labels.append(class_index)
    return images, labels


def augment_batch(batch):
    """Light augmentation: brightness/contrast jitter + small horizontal shift.

    Keeps the model robust to exposure changes and to the user not placing
    the crop box in exactly the same spot every time.
    """
    out = np.empty_like(batch)
    for i, img in enumerate(batch):
        img = img * random.uniform(0.7, 1.3)                    # brightness
        img = (img - 0.5) * random.uniform(0.8, 1.2) + 0.5      # contrast
        shift = random.randint(-8, 8)
        img = np.roll(img, shift, axis=1)                       # horizontal
        out[i] = np.clip(img, 0.0, 1.0)
    return out


def to_tensor(batch_nhwc):
    return torch.from_numpy(batch_nhwc.transpose(0, 3, 1, 2))  # NHWC -> NCHW


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="dataset",
                        help="Folder containing no/, N/ and S/ (default: dataset).")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--out", default="subtitle_detector.onnx",
                        help="Output model path (default: subtitle_detector.onnx).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    folders = [(os.path.join(args.dataset, name), i)
               for i, name in enumerate(CLASS_NAMES)]
    for folder, _ in folders:
        if not os.path.isdir(folder):
            print(f"Error: missing folder {folder} — see TRAINING.md.",
                  file=sys.stderr)
            return 1

    print("Loading dataset...")
    all_x, all_y = [], []
    for folder, class_index in folders:
        images, labels = load_folder(folder, class_index)
        print(f"  {CLASS_NAMES[class_index]}: {len(images)}")
        if len(images) < 50:
            print(f"Warning: fewer than 50 images in '{CLASS_NAMES[class_index]}' "
                  "— expect poor results. Aim for 300+ per class.")
        all_x.extend(images)
        all_y.extend(labels)

    x = np.stack(all_x)
    y = np.array(all_y, dtype=np.int64)

    # Shuffled train/val split.
    order = np.random.permutation(len(x))
    x, y = x[order], y[order]
    n_val = max(1, int(len(x) * args.val_split))
    val_x, val_y = x[:n_val], y[:n_val]
    train_x, train_y = x[n_val:], y[n_val:]
    print(f"  train: {len(train_x)}   val: {len(val_x)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} for {args.epochs} epochs...")
    model = TinyPresenceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    val_xt = to_tensor(val_x).to(device)
    val_yt = torch.from_numpy(val_y).to(device)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_x))
        total_loss = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = to_tensor(augment_batch(train_x[idx])).to(device)
            target = torch.from_numpy(train_y[idx]).to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(idx)

        model.eval()
        with torch.no_grad():
            val_acc = float((model(val_xt).argmax(dim=1) == val_yt).float().mean())
        print(f"  epoch {epoch:2d}: loss={total_loss / len(train_x):.4f} "
              f"val_acc={val_acc * 100:.1f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    print(f"Best validation accuracy: {best_acc * 100:.1f}%")
    model.load_state_dict(best_state)
    model.cpu().eval()

    # The app consumes probabilities, so bake softmax into the exported graph.
    export_model = nn.Sequential(model, nn.Softmax(dim=1)).eval()

    print(f"Exporting to {args.out}...")
    dummy = torch.zeros(1, 3, MODEL_INPUT_H, MODEL_INPUT_W)
    export_kwargs = dict(
        input_names=["image"], output_names=["probs"], opset_version=12,
    )
    try:
        # The newer dynamo-based exporter emits ONNX that cv2.dnn cannot
        # parse (Conv nodes without kernel_size); force the legacy exporter.
        torch.onnx.export(export_model, dummy, args.out, dynamo=False,
                          **export_kwargs)
    except TypeError:
        # Older torch without the `dynamo` argument uses legacy by default.
        torch.onnx.export(export_model, dummy, args.out, **export_kwargs)

    # Sanity check: the app runs the model via cv2.dnn — make sure it agrees
    # with PyTorch on one validation image before declaring victory.
    net = cv2.dnn.readNetFromONNX(args.out)
    sample = val_x[0]
    net.setInput(sample.transpose(2, 0, 1)[np.newaxis])
    cv_probs = net.forward().reshape(-1)
    with torch.no_grad():
        torch_probs = export_model(to_tensor(sample[np.newaxis])).numpy().reshape(-1)
    if np.max(np.abs(cv_probs - torch_probs)) > 1e-3:
        print(f"Warning: cv2.dnn {cv_probs} and torch {torch_probs} disagree "
              "— the exported model may misbehave in the app.")
    else:
        pred = CLASS_NAMES[int(np.argmax(cv_probs))]
        print(f"cv2.dnn check OK (sample -> {pred}, probs "
              + ", ".join(f"{name}={p:.3f}" for name, p in zip(CLASS_NAMES, cv_probs))
              + ").")

    print(f"\nDone. Select {args.out} in the app's model field (or --model).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
