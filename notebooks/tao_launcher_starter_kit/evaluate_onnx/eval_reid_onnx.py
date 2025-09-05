import argparse
import os
import re
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    ort = None


def parse_market1501_dir(dir_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Parse a Market-1501 style directory and return files, pids, cams.

    Expects filenames like: 0001_c1s1_XXXX.jpg, -1 as junk (skipped).
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if os.path.splitext(f)[1].lower() in img_exts
    ]
    pid_list = []
    cam_list = []
    kept_files = []
    pat = re.compile(r"^([\-\d]+)_c(\d+)s\d+_.*\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
    for f in files:
        name = os.path.basename(f)
        m = pat.match(name)
        if not m:
            # Skip files that do not match the expected naming
            continue
        pid_str, cam_str = m.group(1), m.group(2)
        pid = int(pid_str)
        cam = int(cam_str)
        if pid == -1:
            # junk images are ignored
            continue
        kept_files.append(f)
        pid_list.append(pid)
        cam_list.append(cam)
    if not kept_files:
        raise RuntimeError(f"No valid images found in {dir_path}")
    return kept_files, np.array(pid_list, dtype=np.int32), np.array(cam_list, dtype=np.int32)


def get_session(model_path: str, provider_preference: str = "auto"):
    if ort is None:
        raise RuntimeError("onnxruntime is not available. Please install onnxruntime or onnxruntime-gpu.")
    providers = ort.get_available_providers()
    selected = None
    if provider_preference == "cpu":
        selected = ["CPUExecutionProvider"]
    elif provider_preference == "cuda":
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError("CUDAExecutionProvider not available in onnxruntime installation")
        selected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:  # auto
        selected = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=selected)
    return sess


def infer_input_spec(sess, fallback_hw=(256, 128)):
    inp = sess.get_inputs()[0]
    name = inp.name
    shape = list(inp.shape)
    # Expect NCHW. If dynamic, fallback to provided H, W.
    C = int(shape[1]) if isinstance(shape[1], int) else 3
    H = int(shape[2]) if isinstance(shape[2], int) else int(fallback_hw[0])
    W = int(shape[3]) if isinstance(shape[3], int) else int(fallback_hw[1])
    return name, C, H, W


def preprocess_image(path: str, H: int, W: int, mean: Tuple[float, float, float], std: Tuple[float, float, float], color_order: str = "rgb") -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((W, H), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0,1]
    if color_order.lower() == "bgr":
        arr = arr[:, :, ::-1]
    arr = (arr - mean) / std
    chw = np.transpose(arr, (0, 1, 2)).transpose(2, 0, 1)  # HWC->CHW
    return chw


def embed_all(sess, files: List[str], input_name: str, H: int, W: int, batch_size: int,
              mean: Tuple[float, float, float], std: Tuple[float, float, float], color_order: str) -> np.ndarray:
    outs = sess.get_outputs()
    out_name = outs[0].name
    embs = []
    n = len(files)
    for i in range(0, n, batch_size):
        batch_files = files[i:i + batch_size]
        batch = np.stack([preprocess_image(f, H, W, mean, std, color_order) for f in batch_files], axis=0)
        ort_inputs = {input_name: batch.astype(np.float32)}
        out = sess.run([out_name], ort_inputs)[0]
        # L2 normalize
        norm = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        out = out / norm
        embs.append(out)
    return np.vstack(embs)


def cdist_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - A @ B.T


def evaluate_reid(Q: np.ndarray, q_ids: np.ndarray, q_cams: np.ndarray,
                  G: np.ndarray, g_ids: np.ndarray, g_cams: np.ndarray,
                  topk=(1, 5, 10), metric='cosine'):
    if metric == 'cosine':
        dist = cdist_cosine(Q, G)
    else:
        dist = np.sqrt(np.maximum(((Q[:, None, :] - G[None, :, :]) ** 2).sum(axis=2), 1e-12))

    num_q = Q.shape[0]
    cmc = np.zeros(G.shape[0], dtype=np.float64)
    APs = []

    for i in range(num_q):
        order = np.argsort(dist[i])
        # Remove same-id same-camera from gallery
        valid = ~((g_ids[order] == q_ids[i]) & (g_cams[order] == q_cams[i]))
        order = order[valid]
        matches = (g_ids[order] == q_ids[i]).astype(np.int32)

        if matches.sum() == 0:
            continue

        first_hit = np.argmax(matches == 1)
        cmc[first_hit:] += 1

        cumsum = np.cumsum(matches)
        precision = cumsum / (np.arange(len(matches)) + 1)
        AP = (precision * matches).sum() / matches.sum()
        APs.append(AP)

    cmc_curve = cmc / max(1, len(APs))
    res = {'mAP': float(np.mean(APs)) if APs else 0.0}
    for k in topk:
        if len(cmc_curve) == 0:
            res[f'CMC@{k}'] = 0.0
        else:
            res[f'CMC@{k}'] = float(cmc_curve[min(k - 1, len(cmc_curve) - 1)])
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReID ONNX model on Market-1501 style data (CMC/mAP)")
    # Default to the TAO pretrained ONNX (validated). You can override with --model.
    parser.add_argument(
        '--model',
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), 'tao_pretrained_models', 'market1501', 'resnet50_market1501_model.onnx'
        ),
        help='Path to ONNX model file'
    )
    parser.add_argument('--query-dir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'reidentificationnet', 'market1501', 'sample_query'),
                        help='Query directory (Market-1501 format)')
    parser.add_argument('--gallery-dir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'reidentificationnet', 'market1501', 'sample_test'),
                        help='Gallery directory (Market-1501 format)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--provider', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--require-cuda', action='store_true', help='Fail if CUDA EP is not active when provider=cuda')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--height', type=int, default=256, help='Fallback/preferred input height')
    parser.add_argument('--width', type=int, default=128, help='Fallback/preferred input width')
    parser.add_argument('--mean', type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', type=float, nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument('--color-order', type=str, default='rgb', choices=['rgb', 'bgr'])
    args = parser.parse_args()

    # Load data lists
    q_files, q_ids, q_cams = parse_market1501_dir(args.query_dir)
    g_files, g_ids, g_cams = parse_market1501_dir(args.gallery_dir)

    # Create session and infer input spec
    sess = get_session(args.model, provider_preference=args.provider)
    try:
        avail = ort.get_available_providers()
    except Exception:
        avail = []
    active = sess.get_providers()
    print(f"ONNX Runtime providers available: {avail}")
    print(f"ONNX Runtime providers active  : {active}")
    if args.provider == 'cuda' and args.require_cuda and ('CUDAExecutionProvider' not in active):
        raise RuntimeError(
            "CUDAExecutionProvider not active. Check CUDA/cuDNN/cuRAND dependencies or install onnxruntime-gpu."
        )
    inp_name, C, H_model, W_model = infer_input_spec(sess, fallback_hw=(args.height, args.width))
    H, W = H_model, W_model
    if C != 3:
        raise RuntimeError(f"Expected 3-channel input, got C={C}")

    mean = np.array(args.mean, dtype=np.float32)
    std = np.array(args.std, dtype=np.float32)

    # Extract embeddings
    Q = embed_all(sess, q_files, inp_name, H, W, args.batch_size, mean, std, args.color_order)
    G = embed_all(sess, g_files, inp_name, H, W, args.batch_size, mean, std, args.color_order)

    # Evaluate
    res = evaluate_reid(Q, q_ids, q_cams, G, g_ids, g_cams, topk=(1, 5, 10), metric=args.metric)
    print("Results:")
    print(f"mAP   : {res['mAP']:.4f}")
    print(f"CMC@1 : {res['CMC@1']:.4f}")
    print(f"CMC@5 : {res['CMC@5']:.4f}")
    print(f"CMC@10: {res['CMC@10']:.4f}")


if __name__ == '__main__':
    main()
