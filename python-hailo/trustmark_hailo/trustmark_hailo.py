# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


# TrustMark-compatible class for the Hailo port - mirrors the constructor and
# encode()/decode()/schemaCapacity() interface of ../python/trustmark/trustmark.py's
# TrustMark, so code written against the real thing needs minimal changes to
# run against Hailo-accelerated (or CPU/onnxruntime) inference instead:
#
#     from trustmark_hailo import TrustMark
#     tm = TrustMark(model_type='Q')
#     stego = tm.encode(cover_image, "my secret", MODE="text")
#     secret, detected, version = tm.decode(stego, MODE="text")
#

import os
import pathlib
import urllib.request
from hashlib import md5
from mmap import ACCESS_READ, mmap
from typing import Optional

from .pipeline import TrustMarkHailo

# TODO: point at real hosting once the compiled models are actually published -
# this is a local test server for now, serving the finalized .hef files from
# cai-hosted-assets/watermarking/trustmark-models/ (run
# `python3 -m http.server 8000 --bind 0.0.0.0` from that directory). Not
# literally "localhost" here since the Pi is a separate machine on the LAN -
# update this to whatever host is actually running that server.
MODEL_REMOTE_HOST = "http://10.0.0.100:8000/watermarking/trustmark-models/"

# secret2image_{TYPE}.npz is NOT in here - those files are small (~300KB, vs multi-MB
# for the .hef/.onnx below) and ship directly in trustmark_hailo/models/ rather than
# being downloaded on first use - see check_and_download()'s docstring.
# box_head_{TYPE}.npz IS in here (unlike secret2image) - it's ~55MB, too big to
# ship in the package, so it downloads on first use just like the .hef files.
MODEL_CHECKSUMS = {
    "encoder_C.hef": "6fde6c40a0357f9db72d347a6cf45681",  # opt0 (opt2 produced a catastrophically broken model for C specifically, PSNR ~10dB - opt0 validated 95.5% full NPU+NPU round-trip on 200 held-out images, vs 100% for the float ONNX baseline)
    "decoder_C.hef": "5336b95d42c59af6982311e790de57be",  # opt4/ds5120/batch32 - validated exact-match via CPU-encoder+NPU-decoder round trip on real hardware
    "encoder_C.onnx": "e3f9ad8784788ae208bb37a88c1fc50d",
    "decoder_C.onnx": "1a70390e044540de00af2671bc567faf",

    "encoder_Q.hef": "32bf3832f5cf3a4cc7a1a89e9dadf3e5",  # opt2 (finetune) - validated 93.0% full round-trip on 200 held-out images, real Hailo-8L hardware, vs 91.5% for the earlier opt4/ds1024 encoder
    "decoder_Q.hef": "b3fbe8502e5575be64ef991c34409c1a",
    "encoder_Q.onnx": "d6fa92d4b4fe8d67ad196b2d21320955",
    "decoder_Q.onnx": "90427159004ddbf5128271ba0684cf42",
    "bbox_trunk_Q.hef": "2797850f2bd9a743bebf2695d0aad3ff",
    "box_head_Q.npz": "b041ad168a43938e1f6547e075576627",

    "encoder_B.hef": "0214c153ed48abb88aead8d18724f9f0",  # opt2 (finetune) - validated healthy (PSNR 35.3dB, 99.5% exact-match with CPU-decoder reference on 200 held-out images); full NPU+NPU round-trip pending decoder_B's compile
    "decoder_B.hef": "46bb169414e4f473c5d2b5ecdabdbb4c",  # opt4/ds5120/batch32 - validated 99.0% full NPU+NPU round-trip (encoder_B+decoder_B) on 200 held-out images
    "encoder_B.onnx": "4637c2eec9c44d334db833d08299bdce",
    "decoder_B.onnx": "5ff772e2486d286f32e21aced3694b60",

    "encoder_P.hef": "2905023b342c88afb1b290beae8ed7ca",  # not usable on the NPU, see docs - kept for completeness
    "decoder_P.hef": "cdd02c7b9ca687b2ed18b8576d34042a",
    "encoder_P.onnx": "4cb6dfe2ec5e97e7ced942c7c6239e6f",
    "decoder_P.onnx": "2c77408a11bf4f22bb3dcf9e5706dfdd",
    "bbox_trunk_P.hef": "84b2b73550390c12a40a06fa0cd5327a",  # opt2 (was opt0 - reconciled to match Q's recipe). Smoke-tested: IoU 0.9850 vs ground truth, exact decode match, correct None on unwatermarked negative control
    "box_head_P.npz": "8d8619052367bfb2f88bedaef8684572",
}


class TrustMark:
    class Encoding:
        Undefined = -1
        BCH_SUPER = 0
        BCH_5 = 1
        BCH_4 = 2
        BCH_3 = 3

    def __init__(
        self,
        model_type: str = "Q",
        secret_len: int = 100,
        encoding_type: int = Encoding.BCH_5,
        use_ECC: bool = True,
        verbose: bool = True,
        encoder_backend: Optional[str] = None,
        decoder_backend: Optional[str] = "npu",
        loadBBoxDetector: bool = False,
    ):
        assert model_type in ["C", "Q", "B", "P"], "model_type must be one of ['C', 'Q', 'B', 'P']"
        if loadBBoxDetector and model_type not in ("Q", "P"):
            raise ValueError(
                f"loadBBoxDetector=True is only supported for model_type in ('Q', 'P'), got {model_type!r} "
                "- the bbox detector's trained weights aren't published for 'C'/'B', same as upstream."
            )
        if not use_ECC:
            raise NotImplementedError(
                "use_ECC=False is not implemented in the Hailo port - see "
                "../python/trustmark/trustmark.py's raw-bit encode()/subimage_decode() "
                "branches for what this would need to mirror."
            )

        self.model_type = model_type
        self.use_ECC = use_ECC
        self.verbose = verbose
        # Same location as upstream's models: inside the package directory
        # itself, not the caller's cwd - files auto-download here on first use.
        models_dir = pathlib.Path(__file__).parent.resolve() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = models_dir

        if encoder_backend is None:
            encoder_backend = "cpu" if model_type == "P" else "npu"

        if verbose:
            print(
                f"Initializing TrustMark (Hailo port) model_type={model_type!r} "
                f"encoder=[{encoder_backend or 'disabled'}] decoder=[{decoder_backend or 'disabled'}] "
                f"models_dir={models_dir}"
            )

        self._owned_models = []  # HailoModel/OnnxModel instances opened below, released in close()

        encoder_infer = self._make_encoder_infer(models_dir, model_type, encoder_backend) if encoder_backend else None
        decoder_infer = self._make_decoder_infer(models_dir, model_type, decoder_backend) if decoder_backend else None
        bbox_infer = self._make_bbox_infer(models_dir, model_type) if loadBBoxDetector else None

        self._impl = TrustMarkHailo(
            encoder_infer=encoder_infer,
            decoder_infer=decoder_infer,
            secret_len=secret_len,
            encoding_type=encoding_type,
            model_type=model_type,
            bbox_infer=bbox_infer,
        )

    def check_and_download(self, filename):
        valid = False
        if os.path.isfile(filename) and os.path.getsize(filename) > 0:
            with open(filename) as file, mmap(file.fileno(), 0, access=ACCESS_READ) as file:
                valid = MODEL_CHECKSUMS[pathlib.Path(filename).name] == md5(file).hexdigest()

        if not valid:
            if self.verbose:
                print("Fetching model file (once only): " + str(filename))
            urld = MODEL_REMOTE_HOST + os.path.basename(filename)

            urllib.request.urlretrieve(urld, filename=filename)

    def _make_encoder_infer(self, models_dir: pathlib.Path, model_type: str, backend: str):
        if backend == "npu":
            from .hailo_runtime import make_encoder_infer

            hef_path = models_dir / f"encoder_{model_type}.hef"
            # secret2image_{TYPE}.npz ships in trustmark_hailo/models/ already - not
            # downloaded, see MODEL_CHECKSUMS' comment.
            npz_path = models_dir / f"secret2image_{model_type}.npz"
            self.check_and_download(hef_path)
            infer, model = make_encoder_infer(hef_path, npz_path)
        elif backend == "cpu":
            from .onnx_runtime import make_encoder_infer

            onnx_path = models_dir / f"encoder_{model_type}.onnx"
            self.check_and_download(onnx_path)
            infer, model = make_encoder_infer(onnx_path)
        else:
            raise ValueError(f"encoder_backend must be 'npu', 'cpu', or None/False, got {backend!r}")
        self._owned_models.append(model)
        return infer

    def _make_decoder_infer(self, models_dir: pathlib.Path, model_type: str, backend: str):
        if backend == "npu":
            from .hailo_runtime import make_decoder_infer

            hef_path = models_dir / f"decoder_{model_type}.hef"
            self.check_and_download(hef_path)
            infer, model = make_decoder_infer(hef_path)
        elif backend == "cpu":
            from .onnx_runtime import make_decoder_infer

            onnx_path = models_dir / f"decoder_{model_type}.onnx"
            self.check_and_download(onnx_path)
            infer, model = make_decoder_infer(onnx_path)
        else:
            raise ValueError(f"decoder_backend must be 'npu', 'cpu', or None/False, got {backend!r}")
        self._owned_models.append(model)
        return infer

    def _make_bbox_infer(self, models_dir: pathlib.Path, model_type: str):
        from .bbox_runtime import make_bbox_infer

        hef_path = models_dir / f"bbox_trunk_{model_type}.hef"
        # Unlike secret2image_{TYPE}.npz, box_head_{TYPE}.npz is ~55MB - too
        # big to ship in the package, so it downloads on first use too.
        npz_path = models_dir / f"box_head_{model_type}.npz"
        self.check_and_download(hef_path)
        self.check_and_download(npz_path)
        infer, model = make_bbox_infer(hef_path, npz_path)
        self._owned_models.append(model)
        return infer

    def schemaCapacity(self) -> int:
        return self._impl.schemaCapacity()

    def encode(self, in_cover_image, string_secret: str, MODE: str = "text", WM_STRENGTH: float = 1.0):
        return self._impl.encode(in_cover_image, string_secret, MODE=MODE, WM_STRENGTH=WM_STRENGTH)

    def subimage_decode(self, stego_image, MODE: str = "text"):
        return self._impl.subimage_decode(stego_image, MODE=MODE)

    def localize(self, in_stego_image, return_all: bool = False):
        return self._impl.localize(in_stego_image, return_all=return_all)

    def decode(self, in_stego_image, MODE: str = "text", DETECTFIRST: bool = False, ROTATION: bool = False):
        return self._impl.decode(in_stego_image, MODE=MODE, DETECTFIRST=DETECTFIRST, ROTATION=ROTATION)

    def close(self) -> None:
        for model in self._owned_models:
            model.close()
        self._owned_models = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
