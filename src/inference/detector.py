"""
src/inference/detector.py — Inference API for HSF-CVIT
======================================================

Wraps a trained checkpoint behind a clean interface that accepts raw images
or video files and returns a fake/real prediction with confidence score.

Face detection
--------------
The model was trained on face-cropped frames from FaceForensics++.  When
running on uncropped inputs the detector tries to locate the largest face
using OpenCV's Haar cascade (bundled with opencv-python-headless).  If
detection fails or OpenCV is unavailable the full image is used instead.

Usage
-----
    import yaml
    from src.inference.detector import DeepFakeDetector

    with open("configs/train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)

    detector = DeepFakeDetector("outputs/checkpoints/best.pt", train_cfg)

    # Single image (path or PIL.Image)
    result = detector.predict_image("face.jpg")
    # {'label': 'fake', 'probability': 0.87, 'threshold': 0.45, 'face_detected': True}

    # Video file
    result = detector.predict_video("clip.mp4", num_frames=16)
    # {'label': 'fake', 'probability': 0.83, 'frame_probs': [...], 'num_frames_used': 16}

    # Convenience constructor (loads config from disk)
    detector = DeepFakeDetector.from_config_path("outputs/checkpoints/best.pt")
"""

import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from src.models.hsf_cvit import build_model
from src.data.transforms import get_transforms
from src.utils.helpers import extract_frames


class DeepFakeDetector:
    """Load a trained HSF-CVIT checkpoint and run inference on images or videos.

    Parameters
    ----------
    checkpoint_path : str
        Path to a saved ``best.pt`` or ``epoch_NNN.pt`` checkpoint.
    train_cfg : dict
        Loaded ``configs/train_config.yaml``.
    device : str or None
        ``'cuda'``, ``'cpu'``, or None (auto-detect).
    face_margin : float
        Fractional padding added around each detected face box (0.3 = 30 %).
    """

    def __init__(
        self,
        checkpoint_path: str,
        train_cfg: dict,
        device: Union[str, None] = None,
        face_margin: float = 0.3,
    ):
        self.face_margin = face_margin
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        image_size = train_cfg["data"].get("image_size", 224)
        self.transform = get_transforms("val", image_size)
        self._amp = self.device.type == "cuda"

        self.model = build_model(train_cfg)
        self.threshold = 0.5
        self._load_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)

        self._face_detector = self._build_face_detector()

    # ------------------------------------------------------------------
    #  Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config_path(
        cls,
        checkpoint_path: str = "outputs/checkpoints/best.pt",
        config_path: str = "configs/train_config.yaml",
        device: Union[str, None] = None,
        face_margin: float = 0.3,
    ) -> "DeepFakeDetector":
        """Load both the checkpoint and config from file paths."""
        import yaml
        with open(config_path, "r") as f:
            train_cfg = yaml.safe_load(f)
        return cls(checkpoint_path, train_cfg, device=device, face_margin=face_margin)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.threshold = float(ckpt.get("best_threshold", 0.5))
        epoch = ckpt.get("epoch", "?")
        auc   = ckpt.get("best_val_auc", float("nan"))
        print(
            f"  Loaded checkpoint: {path}  "
            f"(epoch {epoch}, val AUC {auc:.4f}, threshold {self.threshold:.2f})"
        )

    def _build_face_detector(self):
        """Return an OpenCV Haar-cascade detector, or None if unavailable."""
        try:
            import cv2
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            if os.path.exists(cascade_path):
                return cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass
        return None

    def _crop_face(self, image: Image.Image) -> tuple[Image.Image, bool]:
        """Detect the largest face and return a margin-padded crop.

        Returns (image, face_was_detected).  Falls back to the original
        image if no face is found or OpenCV is unavailable.
        """
        if self._face_detector is None:
            return image, False

        try:
            import cv2
            arr = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            faces = self._face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                return image, False

            # Pick the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            mw, mh = int(w * self.face_margin), int(h * self.face_margin)
            ih, iw = arr.shape[:2]
            x0 = max(0, x - mw)
            y0 = max(0, y - mh)
            x1 = min(iw, x + w + mw)
            y1 = min(ih, y + h + mh)
            return image.crop((x0, y0, x1, y1)), True
        except Exception:
            return image, False

    def _to_pil(self, image) -> Image.Image:
        """Accept a path, PIL Image, or numpy array and return a PIL RGB Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")

    # ------------------------------------------------------------------
    #  Public inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_image(
        self,
        image: Union[str, Path, "Image.Image", np.ndarray],
        detect_face: bool = True,
    ) -> dict:
        """Run inference on a single image.

        Parameters
        ----------
        image : PIL.Image, file path, or HxWx3 numpy array
        detect_face : bool
            Attempt face detection before classifying.  Set to False when
            the input is already a face crop (as in the training data).

        Returns
        -------
        dict
            label         : 'fake' or 'real'
            probability   : float — P(fake)
            threshold     : float — decision boundary
            face_detected : bool
        """
        pil = self._to_pil(image)

        face_detected = False
        if detect_face:
            pil, face_detected = self._crop_face(pil)

        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.amp.autocast(device_type=self.device.type, enabled=self._amp):
            logit = self.model(tensor)
        prob = float(torch.sigmoid(logit.float()).item())

        return {
            "label": "fake" if prob >= self.threshold else "real",
            "probability": prob,
            "threshold": self.threshold,
            "face_detected": face_detected,
        }

    @torch.no_grad()
    def predict_video(
        self,
        video_path: Union[str, Path],
        num_frames: int = 16,
        detect_face: bool = True,
        aggregation: str = "mean",
    ) -> dict:
        """Run inference on a video by aggregating per-frame predictions.

        Parameters
        ----------
        video_path : str or Path
        num_frames : int
            Number of uniformly-spaced frames to sample from the video.
        detect_face : bool
            Attempt face detection on each sampled frame.
        aggregation : 'mean' | 'max'
            'mean' averages all frame probabilities (default, more robust).
            'max'  takes the worst-case frame (more sensitive to single
            manipulated frames).

        Returns
        -------
        dict
            label           : 'fake' or 'real'
            probability     : float — aggregated P(fake)
            threshold       : float
            frame_probs     : list[float]
            num_frames_used : int
        """
        try:
            frames_rgb = extract_frames(str(video_path), max_frames=num_frames)
        except IOError as exc:
            return {
                "label": "error",
                "probability": float("nan"),
                "threshold": self.threshold,
                "frame_probs": [],
                "num_frames_used": 0,
                "error": str(exc),
            }

        frame_probs = [
            self.predict_image(frame, detect_face=detect_face)["probability"]
            for frame in frames_rgb
        ]

        if not frame_probs:
            return {
                "label": "unknown",
                "probability": 0.5,
                "threshold": self.threshold,
                "frame_probs": [],
                "num_frames_used": 0,
            }

        clip_prob = float(max(frame_probs) if aggregation == "max" else np.mean(frame_probs))

        return {
            "label": "fake" if clip_prob >= self.threshold else "real",
            "probability": clip_prob,
            "threshold": self.threshold,
            "frame_probs": frame_probs,
            "num_frames_used": len(frame_probs),
        }
