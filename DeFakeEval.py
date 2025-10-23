import argparse
import clip  # OpenAI CLIP
import logging
import os
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Local utils
from DataUtils import (
    FakePartsV2DatasetBase,
    collate_skip_none,
    REQUIRED_COLS,
)
from blipmodels import blip_decoder  # BLIP captioner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Dataset wrapper
class FakePartsV2Dataset(FakePartsV2DatasetBase):
    """
    Inherit from FakePartsV2DatasetBase and add dual transforms:
      - clip_transform: for CLIP encoder
      - blip_transform: for BLIP caption generator
    Returns:
      sample: Tensor for CLIP (B,C,H,W)
      label: int
      meta:  dict with the base metadata + a tensor for BLIP under key 'blip_image'
    """

    def __init__(
            self,
            *args,
            clip_transform=None,
            blip_transform=None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_transform = clip_transform
        self.blip_transform = blip_transform

    def __getitem__(self, idx: int):
        # Get the usual (PIL image) via the parent, but we need to reimplement to apply dual transforms.
        if idx < 0:
            idx += len(self)
        abs_path = self._abs_paths[idx]
        label = int(self._labels[idx])
        try:
            with open(abs_path, 'rb') as f:
                b = f.read()
            with Image.open(BytesIO(b)) as im:
                im = im.convert("RGB")
                im.load()
        except Exception as e:
            if self.on_corrupt == "raise":
                raise
            elif self.on_corrupt == "warn":
                log.warning(f"Failed to load image: {abs_path} ({e})")
            return None

        # Apply transforms (fall back to identity if missing)
        img_clip = self.clip_transform(im) if self.clip_transform is not None else im
        img_blip = self.blip_transform(im) if self.blip_transform is not None else transforms.ToTensor()(im)

        meta = self._make_meta(idx, label)
        meta["blip_image"] = img_blip  # store for batching later
        return img_clip, label, meta


# Simple MLP head (if needed)
class LinearHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: Optional[List[int]] = None, num_classes: int = 2):
        super().__init__()
        if hidden is None or len(hidden) == 0:
            self.net = nn.Linear(in_dim, num_classes)
        else:
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ReLU(inplace=True), nn.Dropout(0.5)]
                last = h
            layers += [nn.Linear(last, num_classes)]
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Helpers
def _safe_torch_load(path: str):
    """
    Robust loader for PyTorch 2.6+:
    1) Try weights_only=True (safe).
    2) Try allow-listing known globals with torch.serialization.safe_globals.
    3) Fall back to weights_only=False (trusted checkpoints only).
    """
    if not path or not os.path.exists(path):
        return None

    # 1) Safe attempt
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        pass

    # 2) Allowlist known classes if available
    try:
        from torch.serialization import safe_globals as _safe_globals
        with _safe_globals([clip.model.CLIP, nn.Linear, nn.Sequential]):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        pass

    # 3) Last resort (assumes checkpoint is trusted)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e3:
        log.warning(f"Failed to torch.load('{path}') with both safe and unsafe modes: {e3}")
        return None


def _maybe_dataparallel(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        device_ids = list(range(min(2, torch.cuda.device_count())))  # prefer 2 GPUs
        model = nn.DataParallel(model, device_ids=device_ids)
    return model


def _clip_encode_image(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    m = model.module if isinstance(model, nn.DataParallel) else model
    return m.encode_image(imgs)


def _clip_encode_text(model: nn.Module, toks: torch.Tensor) -> torch.Tensor:
    m = model.module if isinstance(model, nn.DataParallel) else model
    return m.encode_text(toks)


# Core
def get_args():
    p = argparse.ArgumentParser(description="DeFakeEval: batched, multi-GPU evaluation with streaming CSV writes")
    p.add_argument("--data_root", type=str, required=True, help="Root folder of frames/videos")
    p.add_argument("--data_csv", type=str, default=None, help="Optional prebuilt CSV index for the dataset")
    p.add_argument("--done_csv_list", type=str, nargs='*', default=[], help="List of done CSVs to skip samples")
    p.add_argument("--results", type=str, required=True, help="Directory to write results CSV into")
    p.add_argument("--mode", type=str, default="frame", choices=["frame", "video"], help="Dataset mode")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 32 as requested)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--model_name", type=str, default="ViT-B/32", help="CLIP backbone name")
    p.add_argument("--finetuned_clip", type=str, default="finetune_clip.pt", help="Optional finetuned CLIP *.pt")
    p.add_argument("--linear_head", type=str, default="clip_linear.pt", help="Classifier head checkpoint (*.pt)")
    p.add_argument("--image_size", type=int, default=224, help="Input size for BLIP")
    p.add_argument("--no_amp", action="store_true", help="Disable autocast mixed precision")
    p.add_argument("--blip_beams", type=int, default=3, help="BLIP num beams (greedy if 1)")
    p.add_argument("--blip_max_len", type=int, default=60, help="BLIP max caption length")
    p.add_argument("--blip_min_len", type=int, default=5, help="BLIP min caption length")
    return p.parse_args()


def build_models(device_main: torch.device, image_size: int, model_name: str):
    # CLIP feature extractor + preprocess (normalisation for CLIP)
    clip_model, clip_preprocess = clip.load(model_name, device="cpu", jit=False)
    # We'll move to GPU(s) later

    # BLIP captioner (kept on a single device to avoid DP issues in generate())
    blip_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    blip = blip_decoder(pretrained=blip_url, image_size=image_size, vit="base").to(device_main).eval()

    # A simple transform for BLIP that matches typical training (no CLIP normalisation)
    blip_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    return clip_model, clip_preprocess, blip, blip_transform


def load_linear_head(path: str, in_dim: int = 1024, num_classes: int = 2) -> nn.Module:
    ckpt = _safe_torch_load(path) if path else None
    if ckpt is None:
        # Fallback
        return LinearHead(in_dim=in_dim, hidden=[512, 256], num_classes=num_classes)

    # If it's a full module saved directly
    if isinstance(ckpt, nn.Module):
        return ckpt

    # If it's a state dict or contains one
    state = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            # Likely a plain state_dict
            state = ckpt

    head = LinearHead(in_dim=in_dim, hidden=[512, 256], num_classes=num_classes)
    if state is not None:
        try:
            head.load_state_dict(state, strict=False)
        except Exception as e:
            log.warning(f"Could not load linear head state_dict strictly: {e}")
    return head


def stream_write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    df = pd.DataFrame(rows, columns=list(REQUIRED_COLS))
    # Create directory if needed
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Append with header only if file doesn't exist
    write_header = not out_csv.exists()
    df.to_csv(out_csv, mode="a", header=write_header, index=False)


def main():
    args = get_args()

    torch.backends.cudnn.benchmark = True  # speed-up
    use_amp = not args.no_amp

    # Devices
    device_main = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Models
    log.info("Building models...")
    clip_model, clip_preprocess, blip, blip_transform = build_models(
        device_main=device_main, image_size=args.image_size, model_name=args.model_name
    )

    # Optionally load finetuned CLIP checkpoint
    if args.finetuned_clip and os.path.exists(args.finetuned_clip):
        ckpt = _safe_torch_load(args.finetuned_clip)
        if ckpt is not None:
            try:
                # Try various common formats
                if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                    clip_model.load_state_dict(ckpt["state_dict"], strict=False)
                elif isinstance(ckpt, dict):
                    clip_model.load_state_dict(ckpt, strict=False)
                else:
                    # If somehow a full model was saved, try to extract state_dict
                    clip_model.load_state_dict(getattr(ckpt, "state_dict", lambda: {})(), strict=False)
            except Exception as e:
                log.warning(f"Could not load finetuned CLIP from {args.finetuned_clip}: {e}")
        else:
            log.warning(f"finetuned CLIP checkpoint not loaded: {args.finetuned_clip}")

    # Linear head
    linear_head = load_linear_head(args.linear_head)

    # Move to GPU(s)
    clip_model = clip_model.eval().to(device_main)
    linear_head = linear_head.eval().to(device_main)

    # DataParallel for CLIP + head (uses up to 2 GPUs if visible)
    clip_model = _maybe_dataparallel(clip_model)
    linear_head = _maybe_dataparallel(linear_head)

    # Dataset & Loader
    log.info("Building dataset and dataloader...")
    dataset = FakePartsV2Dataset(
        data_root=Path(args.data_root),
        mode=args.mode,
        csv_path=args.data_csv,
        model_name=f"DeFake_{args.model_name.replace('/', '')}",
        clip_transform=clip_preprocess,  # CLIP-normalised tensors
        blip_transform=blip_transform,  # BLIP tensors
        on_corrupt="warn",
        done_csv_list=args.done_csv_list,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_skip_none,
        drop_last=False,
    )
    log.info(f"Dataset size: {len(dataset)} samples; DataLoader ready.")
    log.info(f"Batch size: {args.batch_size}; batches: {len(loader)}; workers: {args.num_workers}")

    # Output path
    results_dir = Path(args.results)
    out_csv = results_dir / "predictions.csv"
    log.info(f"Writing streaming results to: {out_csv}")
    log.info(f"Dataset: {dataset}")
    log.info(f"Batch size: {args.batch_size}; workers: {args.num_workers}; GPUs: {torch.cuda.device_count()}")

    # Evaluation loop
    rows: List[Dict[str, Any]] = []
    log.info("Starting evaluation...")
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Evaluating"):
        if batch is None:
            continue  # all samples in this batch were None/corrupt
        imgs_clip, labels, metas = batch  # imgs_clip: (B,C,H,W)
        B = imgs_clip.size(0)

        # Prepare BLIP images: list of tensors -> stack
        blip_imgs_list = metas["blip_image"]
        imgs_blip = torch.stack(blip_imgs_list, dim=0)

        # Devices
        imgs_clip = imgs_clip.to(device_main, non_blocking=True)
        # BLIP kept on device_main (cuda:0 or cpu)
        imgs_blip = imgs_blip.to(device_main, non_blocking=True)

        # Generate captions (batched)
        with torch.inference_mode():
            captions: List[str] = blip.generate(
                imgs_blip,
                sample=False if args.blip_beams == 1 else True,
                num_beams=args.blip_beams,
                max_length=args.blip_max_len,
                min_length=args.blip_min_len,
            )
        # Tokenise for CLIP
        text_tokens = clip.tokenize(captions, truncate=True).to(device_main, non_blocking=True)

        # Encode & classify
        with torch.cuda.amp.autocast(enabled=use_amp), torch.inference_mode():
            img_feats = _clip_encode_image(clip_model, imgs_clip)
            txt_feats = _clip_encode_text(clip_model, text_tokens)
            # Ensure shapes align
            if img_feats.dim() == 1:
                img_feats = img_feats.unsqueeze(0)
            if txt_feats.dim() == 1:
                txt_feats = txt_feats.unsqueeze(0)
            emb = torch.cat([img_feats, txt_feats], dim=1).float()
            logits = linear_head(emb)  # (B,2)
            probs = F.softmax(logits, dim=1)  # class 1 -> fake score
            pred = torch.argmax(logits, dim=1).detach().cpu().tolist()
            score = probs[:, 1].detach().cpu().tolist()

        # Build rows in REQUIRED_COLS order
        for i in range(B):
            rows.append({
                "sample_id": metas["sample_id"][i],
                "task": metas["task"][i],
                "method": metas["method"][i],
                "subset": metas["subset"][i],
                "label": int(labels[i]),
                "model": dataset.model_name,
                "mode": metas["mode"][i],
                "score": float(score[i]),
                "pred": int(pred[i]),
            })

        # Stream-write every batch to avoid data loss on interruption
        stream_write_csv(rows, out_csv)
        rows.clear()  # clear buffer

        if (batch_idx + 1) % 10 == 0:
            tqdm.write(f"Processed {batch_idx + 1} batches...")

    log.info(f"[done] Finished. Results saved to: {out_csv}")


if __name__ == "__main__":
    main()
