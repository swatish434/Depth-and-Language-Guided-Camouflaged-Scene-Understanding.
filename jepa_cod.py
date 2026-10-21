#!/usr/bin/env python3
"""
JEPA-BAD: I-JEPA encoder + Boundary-Aware Decoder for COD/COS.
No py_sod_metrics required. Test sets: CAMO, COD10K, CHAMELEON (no NC4K).
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 14
TEST_SETS = ("CAMO", "COD10K", "CHAMELEON")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class CODDataset(Dataset):
    def __init__(self, img_dir, gt_dir, size=448, train=True):
        self.img_dir = Path(img_dir)
        self.gt_dir = Path(gt_dir)
        self.size = size
        self.train = train
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"GT dir not found: {self.gt_dir}")

        self.imgs = sorted(
            p for p in self.img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        kept = []
        for p in self.imgs:
            if self._find_gt(p.stem) is not None:
                kept.append(p)
        missing = len(self.imgs) - len(kept)
        if missing:
            print(f"[WARN] {self.img_dir}: skipped {missing} images without GT")
        self.imgs = kept
        if not self.imgs:
            raise RuntimeError(f"No image/GT pairs under {img_dir} / {gt_dir}")

    def __len__(self):
        return len(self.imgs)

    def _find_gt(self, stem):
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            q = self.gt_dir / f"{stem}{ext}"
            if q.exists():
                return q
        return None

    def gt_path_for_name(self, name: str) -> Path:
        q = self._find_gt(name)
        if q is None:
            raise FileNotFoundError(f"GT for stem={name} in {self.gt_dir}")
        return q

    @staticmethod
    def _aug(img, gt):
        h, w = gt.shape
        s = math.sqrt(random.uniform(0.5, 1.0))
        ch, cw = max(1, int(h * s)), max(1, int(w * s))
        y = random.randint(0, max(0, h - ch))
        x = random.randint(0, max(0, w - cw))
        img, gt = img[y:y + ch, x:x + cw], gt[y:y + ch, x:x + cw]
        if random.random() < 0.5:
            img, gt = img[:, ::-1], gt[:, ::-1]
        img = np.clip(
            img.astype(np.float32) * random.uniform(0.75, 1.25) + random.uniform(-20, 20),
            0, 255,
        )
        return np.ascontiguousarray(img), np.ascontiguousarray(gt)

    def __getitem__(self, i):
        p = self.imgs[i]
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt_path = self._find_gt(p.stem)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise RuntimeError(f"Failed to read GT {gt_path}")
        h0, w0 = gt.shape[:2]
        if self.train:
            img, gt = self._aug(img, gt)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        gt_r = cv2.resize(gt, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return {
            "image": torch.from_numpy(img).permute(2, 0, 1).float(),
            "mask": torch.from_numpy((gt_r > 127).astype(np.float32))[None],
            "name": p.stem,
            "orig_size": torch.tensor([h0, w0], dtype=torch.long),
            "gt_path": str(gt_path),
        }


def mask_to_edge(mask, k=7):
    pad = k // 2
    dil = F.max_pool2d(mask, k, 1, pad)
    ero = -F.max_pool2d(-mask, k, 1, pad)
    return (dil - ero).clamp(0, 1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class JEPAEncoder(nn.Module):
    def __init__(
        self,
        name="facebook/ijepa_vith14_1k",
        layers=(7, 15, 23, 31),
        unfreeze_last=0,
        grad_ckpt=True,
    ):
        super().__init__()
        try:
            from transformers import IJepaModel
            self.vit = IJepaModel.from_pretrained(name)
        except Exception:
            from transformers import AutoModel
            self.vit = AutoModel.from_pretrained(name, trust_remote_code=True)

        self.layers = tuple(layers)
        cfg = self.vit.config
        self.patch = int(getattr(cfg, "patch_size", PATCH))
        self.embed_dim = int(getattr(cfg, "hidden_size", 1280))
        depth = int(getattr(cfg, "num_hidden_layers", 32))

        for p in self.vit.parameters():
            p.requires_grad = False
        self.unfreeze_last = int(unfreeze_last)

        blocks = None
        if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
            blocks = self.vit.encoder.layer
        elif hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layers"):
            blocks = self.vit.encoder.layers
        elif hasattr(self.vit, "blocks"):
            blocks = self.vit.blocks

        if self.unfreeze_last > 0 and blocks is not None:
            for blk in list(blocks)[depth - self.unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True
            if grad_ckpt:
                try:
                    self.vit.gradient_checkpointing_enable()
                except Exception:
                    pass

    def forward(self, x):
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        ctx = torch.no_grad() if self.unfreeze_last == 0 else torch.enable_grad()
        with ctx:
            try:
                out = self.vit(
                    pixel_values=x,
                    output_hidden_states=True,
                    interpolate_pos_encoding=True,
                )
            except TypeError:
                out = self.vit(pixel_values=x, output_hidden_states=True)

        hs = out.hidden_states
        maps = []
        for i in self.layers:
            t = hs[i + 1]
            if t.shape[1] == h * w + 1:
                t = t[:, 1:, :]
            elif t.shape[1] != h * w:
                t = t[:, -h * w:, :]
            maps.append(t.transpose(1, 2).reshape(B, -1, h, w))
        return maps


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def cbr(i, o, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(i, o, k, s, k // 2, bias=False),
        nn.GroupNorm(8, o),
        nn.GELU(),
    )


def up_to(x, ref_or_size):
    size = ref_or_size if isinstance(ref_or_size, (tuple, list)) else ref_or_size.shape[-2:]
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DetailStem(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.s1 = nn.Sequential(cbr(3, 32, 3, 2), cbr(32, 32))
        self.s2 = nn.Sequential(cbr(32, c, 3, 2), cbr(c, c))
        self.s3 = nn.Sequential(cbr(c, c, 3, 2), cbr(c, c))

    def forward(self, img):
        f2 = self.s1(img)
        f4 = self.s2(f2)
        f8 = self.s3(f4)
        return f4, f8


class RAStage(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fuse = nn.Sequential(cbr(d, d), cbr(d, d))
        self.pred = nn.Sequential(cbr(d, d // 2), nn.Conv2d(d // 2, 1, 1))

    def forward(self, x, prev=None):
        if prev is None:
            return self.pred(self.fuse(x))
        prev = up_to(prev, x)
        x = x * (1.0 - torch.sigmoid(prev))
        return self.pred(self.fuse(x)) + prev


class BAFM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fuse = cbr(d + 1, d)

    def forward(self, x, edge_logit):
        e = up_to(edge_logit, x)
        x = x * (1.0 + torch.sigmoid(e))
        return self.fuse(torch.cat([x, e], 1))


class FullResRefine(nn.Module):
    def __init__(self, c=24):
        super().__init__()
        self.net = nn.Sequential(cbr(4, c), cbr(c, c), nn.Conv2d(c, 1, 3, 1, 1))

    def forward(self, img, logit):
        logit_up = up_to(logit, img)
        return logit_up + self.net(torch.cat([img, torch.sigmoid(logit_up)], 1))


class JEPABoundaryAwareCOD(nn.Module):
    def __init__(self, encoder, d=256, d8=128, d4=64):
        super().__init__()
        self.enc = encoder
        C = encoder.embed_dim
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(C, d, 1, bias=False),
                    nn.GroupNorm(8, d),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        self.stem = DetailStem(d4)
        self.ra = nn.ModuleList([RAStage(d) for _ in range(4)])
        self.e8 = cbr(d + d4, d8)
        self.e4 = cbr(d8 + d4, d4)
        self.edge_head8 = nn.Conv2d(d8, 1, 1)
        self.edge_head4 = nn.Sequential(cbr(d4, d4), nn.Conv2d(d4, 1, 1))
        self.up8 = cbr(d, d8)
        self.mix8 = cbr(d8 + d4, d8)
        self.bafm8 = BAFM(d8)
        self.ra8 = RAStage(d8)
        self.up4 = cbr(d8, d4)
        self.mix4 = cbr(d4 + d4, d4)
        self.bafm4 = BAFM(d4)
        self.ra4 = RAStage(d4)
        self.refine = FullResRefine()

    def forward(self, img):
        feats = [p(f) for p, f in zip(self.proj, self.enc(img))]
        f4, f8 = self.stem(img)
        H, W = img.shape[-2:]
        aux = []
        logit, g = None, None
        for k in (3, 2, 1, 0):
            x = feats[k] if g is None else feats[k] + g
            logit = self.ra[3 - k](x, logit)
            g = x
            aux.append(logit)
        e8 = self.e8(torch.cat([up_to(feats[0], f8), f8], 1))
        edge8 = self.edge_head8(e8)
        e4 = self.e4(torch.cat([up_to(e8, f4), f4], 1))
        edge4 = self.edge_head4(e4)
        r8 = self.up8(up_to(g, f8))
        r8 = self.bafm8(self.mix8(torch.cat([r8, f8], 1)), edge8)
        logit8 = self.ra8(r8, logit)
        aux.append(logit8)
        r4 = self.up4(up_to(r8, f4))
        r4 = self.bafm4(self.mix4(torch.cat([r4, f4], 1)), edge4)
        logit4 = self.ra4(r4, logit8)
        aux.append(logit4)
        final = self.refine(img, logit4)
        return {
            "mask": final,
            "aux": [up_to(a, (H, W)) for a in aux],
            "edge": [up_to(edge8, (H, W)), up_to(edge4, (H, W))],
        }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def structure_loss(logit, mask):
    w = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    wbce = (
        w * F.binary_cross_entropy_with_logits(logit, mask, reduction="none")
    ).sum((2, 3)) / w.sum((2, 3)).clamp_min(1e-6)
    p = torch.sigmoid(logit)
    inter = (p * mask * w).sum((2, 3))
    union = ((p + mask) * w).sum((2, 3))
    return (wbce + (1 - (inter + 1) / (union - inter + 1))).mean()


def edge_loss(logit, edge):
    pos = edge.sum() + 1
    neg = edge.numel() - edge.sum() + 1
    bce = F.binary_cross_entropy_with_logits(
        logit, edge, pos_weight=(neg / pos).clamp(max=20)
    )
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * edge).sum() + 1) / (p.sum() + edge.sum() + 1)
    return bce + dice


def total_loss(out, mask, lam_aux=0.3, lam_edge=1.0):
    edge_gt = mask_to_edge(mask)
    l_main = structure_loss(out["mask"], mask)
    l_aux = sum(structure_loss(a, mask) for a in out["aux"]) / max(len(out["aux"]), 1)
    l_edge = sum(edge_loss(e, edge_gt) for e in out["edge"]) / max(len(out["edge"]), 1)
    return l_main + lam_aux * l_aux + lam_edge * l_edge, {
        "main": float(l_main.item()),
        "aux": float(l_aux.item()),
        "edge": float(l_edge.item()),
    }


# ---------------------------------------------------------------------------
# Metrics (no py_sod_metrics)
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self):
        self._preds = []
        self._gts = []

    def step(self, pred01, gt01):
        p = np.clip(np.asarray(pred01, dtype=np.float64), 0.0, 1.0)
        g = np.clip(np.asarray(gt01, dtype=np.float64), 0.0, 1.0)
        if p.shape != g.shape:
            return
        self._preds.append(p)
        self._gts.append(g)

    @staticmethod
    def _s_object(pred, gt):
        fg = gt > 0.5
        bg = ~fg
        if fg.sum() == 0:
            return 1.0 - pred.mean()
        if bg.sum() == 0:
            return pred.mean()
        u_fg = pred[fg].mean()
        u_bg = pred[bg].mean()
        s_fg = 2 * u_fg / (u_fg + 1.0 + 1e-8)
        s_bg = 2 * (1.0 - u_bg) / ((1.0 - u_bg) + 1.0 + 1e-8)
        return 0.5 * (s_fg + s_bg)

    @staticmethod
    def _s_region(pred, gt):
        h, w = pred.shape
        ph, pw = max(h // 2, 1), max(w // 2, 1)
        score, n = 0.0, 0
        for y0, y1 in ((0, ph), (ph, h)):
            for x0, x1 in ((0, pw), (pw, w)):
                score += 1.0 - abs(pred[y0:y1, x0:x1].mean() - gt[y0:y1, x0:x1].mean())
                n += 1
        return score / max(n, 1)

    @staticmethod
    def _structure_measure(pred, gt, alpha=0.5):
        return alpha * Metrics._s_object(pred, gt) + (1.0 - alpha) * Metrics._s_region(pred, gt)

    @staticmethod
    def _weighted_f(pred, gt, beta2=0.3):
        g = (gt > 0.5).astype(np.float64)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        band = cv2.dilate(g.astype(np.uint8), k) - cv2.erode(g.astype(np.uint8), k)
        w = 1.0 + 3.0 * band.astype(np.float64)
        pred = np.clip(pred, 0, 1)
        tp = (pred * g * w).sum()
        prec = tp / ((pred * w).sum() + 1e-8)
        rec = tp / ((g * w).sum() + 1e-8)
        return (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-8)

    def get(self):
        if not self._preds:
            return {"Sm": 0.0, "Em": float("nan"), "wFm": 0.0, "MAE": 1.0, "IoU": 0.0, "Dice": 0.0}
        sms, wfms, maes, ious, dices = [], [], [], [], []
        for p, g in zip(self._preds, self._gts):
            maes.append(float(np.mean(np.abs(p - g))))
            pb, gb = p > 0.5, g > 0.5
            inter = float(np.logical_and(pb, gb).sum())
            union = float(np.logical_or(pb, gb).sum()) + 1e-6
            ious.append(inter / union)
            dices.append(2 * inter / (float(pb.sum() + gb.sum()) + 1e-6))
            sms.append(float(self._structure_measure(p, g)))
            try:
                wfms.append(float(self._weighted_f(p, g)))
            except Exception:
                prec = inter / (float(pb.sum()) + 1e-6)
                rec = inter / (float(gb.sum()) + 1e-6)
                beta2 = 0.3
                wfms.append((1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-8))
        return {
            "Sm": float(np.mean(sms)),
            "Em": float("nan"),
            "wFm": float(np.mean(wfms)),
            "MAE": float(np.mean(maes)),
            "IoU": float(np.mean(ious)),
            "Dice": float(np.mean(dices)),
        }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, img, scales=(1.0, 1.25), flip=True):
    H, W = img.shape[-2:]
    pm, pe, n = 0, 0, 0
    for s in scales:
        sz = max(PATCH, int(round(H * s / PATCH)) * PATCH)
        x = F.interpolate(img, size=(sz, sz), mode="bilinear", align_corners=False)
        for fl in ([False, True] if flip else [False]):
            xi = x.flip(-1) if fl else x
            out = model(xi)
            m = torch.sigmoid(out["mask"])
            e = torch.sigmoid(out["edge"][-1])
            if fl:
                m, e = m.flip(-1), e.flip(-1)
            pm = pm + up_to(m, (H, W))
            pe = pe + up_to(e, (H, W))
            n += 1
    return pm / n, pe / n


def mask_to_boxes(prob, thr=0.5, min_area=64):
    binm = (prob > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            boxes.append({
                "xyxy": [int(x), int(y), int(x + w), int(y + h)],
                "score": float(prob[lab == i].mean()),
            })
    return sorted(boxes, key=lambda b: -b["score"])


@torch.no_grad()
def evaluate(model, loader, dev, save_dir=None, tta=True):
    model.eval()
    met, det = Metrics(), {}
    if save_dir:
        save_dir = Path(save_dir)
        for sub in ("mask", "edge"):
            (save_dir / sub).mkdir(parents=True, exist_ok=True)

    for b in tqdm(loader, leave=False, desc="eval"):
        img = b["image"].to(dev, non_blocking=True)
        name = b["name"][0]
        h, w = int(b["orig_size"][0, 0]), int(b["orig_size"][0, 1])

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            pm, pe = predict(
                model, img,
                scales=(1.0, 1.25) if tta else (1.0,),
                flip=tta,
            )

        pm = up_to(pm.float(), (h, w))[0, 0].cpu().numpy()
        pe = up_to(pe.float(), (h, w))[0, 0].cpu().numpy()

        gt_path = b["gt_path"][0] if "gt_path" in b else str(loader.dataset.gt_path_for_name(name))
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"[WARN] missing GT {gt_path}")
            continue
        if gt.shape[0] != h or gt.shape[1] != w:
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        gt01 = (gt > 127).astype(np.float32)

        met.step(pm, gt01)
        det[name] = mask_to_boxes(pm)
        if save_dir:
            cv2.imwrite(str(save_dir / "mask" / f"{name}.png"), (pm * 255).astype(np.uint8))
            cv2.imwrite(str(save_dir / "edge" / f"{name}.png"), (pe * 255).astype(np.uint8))

    if save_dir:
        json.dump(det, open(save_dir / "boxes.json", "w"), indent=2)
    return met.get()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def resolve_train_dirs(args):
    if args.train_img and args.train_gt:
        return Path(args.train_img), Path(args.train_gt)
    root = Path(args.data_root)
    return root / "TrainDataset" / "Imgs", root / "TrainDataset" / "GT"


def resolve_test_dir(data_root: Path, name: str):
    base = data_root / "TestDataset" / name
    img, gt = base / "Imgs", base / "GT"
    if img.is_dir() and gt.is_dir():
        return img, gt
    return None, None


# ---------------------------------------------------------------------------
# Train / test
# ---------------------------------------------------------------------------
def build_model(args, dev):
    enc = JEPAEncoder(args.encoder, unfreeze_last=args.unfreeze_last)
    return JEPABoundaryAwareCOD(enc).to(dev)


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_img, train_gt = resolve_train_dirs(args)
    print(f"Train Imgs: {train_img}")
    print(f"Train GT  : {train_gt}")
    tr = CODDataset(train_img, train_gt, args.size, train=True)
    print(f"Train pairs: {len(tr)}")
    tl = DataLoader(
        tr, args.bs, shuffle=True, num_workers=args.workers,
        drop_last=True, pin_memory=True,
    )

    root = Path(args.data_root)
    val_img, val_gt = resolve_test_dir(root, args.val_set)
    if val_img is None:
        raise FileNotFoundError(
            f"Val set {args.val_set} missing under {root}/TestDataset/{args.val_set}/{{Imgs,GT}}"
        )
    val = CODDataset(val_img, val_gt, args.size, train=False)
    vl = DataLoader(val, 1, shuffle=False, num_workers=min(4, args.workers))

    model = build_model(args, dev)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        print(f"Resumed {args.resume}")

    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    dec_params = [
        p for n, p in model.named_parameters()
        if (not n.startswith("enc.")) and p.requires_grad
    ]
    groups = [{"params": dec_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr * 0.05})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    total_steps = max(1, args.epochs * len(tl))
    warm = max(1, len(tl))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, step = -1.0, 0

    for ep in range(args.epochs):
        model.train()
        if args.unfreeze_last == 0:
            model.enc.eval()
        pbar = tqdm(tl, desc=f"ep{ep}")
        for b in pbar:
            progress = step / total_steps
            warm_f = min(1.0, (step + 1) / warm)
            lr_f = warm_f * 0.5 * (1.0 + math.cos(math.pi * progress))
            for g, l0 in zip(opt.param_groups, base_lrs):
                g["lr"] = l0 * lr_f

            img = b["image"].to(dev, non_blocking=True)
            m = b["mask"].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                o = model(img)
            o_f = {
                k: ([t.float() for t in v] if isinstance(v, list) else v.float())
                for k, v in o.items()
            }
            loss, parts = total_loss(o_f, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", **{k: f"{v:.3f}" for k, v in parts.items()})

        # always keep last.pt
        torch.save({"model": model.state_dict(), "args": vars(args)}, out / "last.pt")

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            try:
                r = evaluate(model, vl, dev, tta=False)
                print(f"ep{ep} {args.val_set}-val", r)
                score = float(r.get("Sm", -1.0))
                if score > best:
                    best = score
                    torch.save(
                        {"model": model.state_dict(), "args": vars(args), "metrics": r},
                        out / "best.pt",
                    )
                    print(f"  saved best Sm={best:.4f}")
            except Exception as e:
                print(f"[WARN] eval failed ep{ep}: {e}")
                torch.save(
                    {"model": model.state_dict(), "args": vars(args)},
                    out / f"epoch{ep}.pt",
                )

    print("best Sm", best)
    if best < 0 and (out / "last.pt").exists():
        print("[INFO] no best.pt from metrics; last.pt is available for resume")


def test(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, dev)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    root = Path(args.data_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    res = {}
    names = args.test_sets if args.test_sets else list(TEST_SETS)

    for name in names:
        img_dir, gt_dir = resolve_test_dir(root, name)
        if img_dir is None:
            print(f"[SKIP] {name}: missing {root}/TestDataset/{name}/{{Imgs,GT}}")
            continue
        ds = CODDataset(img_dir, gt_dir, args.size, train=False)
        print(f"{name}: {len(ds)} images")
        loader = DataLoader(ds, 1, shuffle=False, num_workers=min(4, args.workers))
        metrics = evaluate(
            model, loader, dev, save_dir=out_root / name, tta=not args.no_tta
        )
        res[name] = metrics
        print(name, metrics)

    json.dump(res, open(out_root / "metrics.json", "w"), indent=2)
    print("Wrote", out_root / "metrics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--train_img", default="")
    ap.add_argument("--train_gt", default="")
    ap.add_argument("--val_set", default="CAMO", choices=list(TEST_SETS))
    ap.add_argument("--test_sets", nargs="+", default=None)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--encoder", default="facebook/ijepa_vith14_1k")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--unfreeze_last", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--resume", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_tta", action="store_true")
    args = ap.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if not args.ckpt:
            raise SystemExit("test mode requires --ckpt")
        test(args)


if __name__ == "__main__":
    main()
