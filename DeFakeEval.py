import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

import clip
from tqdm import tqdm

from blipmodels import blip_decoder

from DataUtils import (
    index_dataframe,
    standardise_predictions,
    REQUIRED_COLS,
)  # schema + indexing helpers


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


def load_blip(device, image_size=224):
    url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    model = blip_decoder(pretrained=url, image_size=image_size, vit="base").to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_one(img_path, device, blip, clip_model, clip_preprocess, linear_head):
    # Load image for BLIP captioner
    img_blip = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
    )(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    # Caption
    caption = blip.generate(img_blip, sample=False, num_beams=3, max_length=60, min_length=5)
    text_tokens = clip.tokenize(list(caption)).to(device)

    # CLIP preprocess for image encoder
    img_clip = clip_preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    # Encode
    image_features = clip_model.encode_image(img_clip)
    text_features = clip_model.encode_text(text_tokens)

    # Concatenate and classify
    emb = torch.cat((image_features, text_features), dim=1).float()
    logits = linear_head(emb)  # shape [1, 2] for [real, fake]
    probs = F.softmax(logits, dim=1)  # probability over {real(0), fake(1)}
    fake_prob = float(probs[0, 1].item())
    pred = int(torch.argmax(logits, dim=1).item())  # 0=real, 1=fake

    return fake_prob, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--results", required=True, type=str)
    # Optional: paths/names so you can swap checkpoints easily
    ap.add_argument("--clip_name", default="ViT-B/32", type=str)
    ap.add_argument("--clip_model_ckpt", default="finetune_clip.pt", type=str)  # if you finetuned CLIP
    ap.add_argument("--linear_head_ckpt", default="clip_linear.pt", type=str)  # 2-class head
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results, exist_ok=True)
    out_csv = str(Path(args.results) / "predictions.csv")

    # Index dataset -> DataFrame with task/method/subset/label/mode/rel_path/abs_path/root
    df_idx = index_dataframe(
        Path(args.data_root))  # collects images under data_root
    if len(df_idx) == 0:
        raise SystemExit(f"No files found under {args.data_root}")

    # Load models
    blip = load_blip(device)
    clip_model, clip_preprocess = clip.load(args.clip_name, device=device)

    # Optionally load your finetuned CLIP backbone (if it’s a full nn.Module checkpoint)
    if os.path.exists(args.clip_model_ckpt):
        try:
            clip_model = torch.load(args.clip_model_ckpt, map_location=device, weights_only=False)
        except Exception:
            pass  # if not compatible, silently use the original CLIP

    # Load linear head (expects input dim = image_feat_dim + text_feat_dim)
    linear_head = torch.load(args.linear_head_ckpt, map_location=device, weights_only=False)
    linear_head.eval()
    linear_head.to(device)

    rows = []
    for _, r in tqdm(df_idx.iterrows(), total=len(df_idx), desc="Processing images"):
        img_path = r["abs_path"]
        try:
            score, pred = infer_one(img_path, device, blip, clip_model, clip_preprocess, linear_head)
        except Exception:
            # On any failure, emit a neutral row with score=-1 but keep schema intact
            score, pred = -1.0, 0

        # Build REQUIRED_COLS row; higher score must mean "more fake"
        rows.append({
            "sample_id": r["rel_path"],  # stable join key
            "task": r["task"],
            "method": r["method"],
            "subset": r["subset"],
            "label": int(r["label"]),  # 0=real, 1=fake (ground truth)
            "model": f"DeFake-{args.clip_name}",  # simple identifier
            "mode": r["mode"],  # 'video' or 'frame'
            "score": float(score),  # P(fake)
            "pred": int(pred),  # argmax over {real(0), fake(1)}
        })

    # Standardise and save (validates REQUIRED_COLS, coerces dtypes)
    df_out = standardise_predictions(rows)  # ensures schema and types
    # Sanity: keep only REQUIRED_COLS, preserve order
    df_out = df_out.loc[:, list(REQUIRED_COLS)]  # ordered columns
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
