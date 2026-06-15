"""VisualBERT classifier wrapper for one-shot binary misogyny classification.

Uses ``uclanlp/visualbert-vqa-coco-pre`` (COCO-pretrained VisualBERT with an MLM head) via
``VisualBertForPreTraining`` to perform a **masked-language-model cloze** zero-shot
classification on meme images.

Approach (MLM cloze):
    The same instruction shown to the generative VLMs is used, followed by a single
    ``[MASK]`` token in the one-word answer slot, e.g.::

        "Is this meme misogynistic? ... Answer with exactly one word: yes or no. [MASK]"

    The model predicts a distribution over the full BERT vocabulary at the ``[MASK]``
    position.  We compare the logits for the tokens ``"yes"`` and ``"no"`` and pick the
    higher one.  This is an **untrained zero-shot baseline expected to be near-random**
    on MAMI; it is included to quantify how much language-prior alone contributes.

Visual features are extracted with a pretrained ResNet-50 (final fc layer removed),
following the approach described in the HuggingFace VisualBERT documentation.
The text is tokenized via ``google-bert/bert-base-uncased``.

Both singleclass (binary yes/no) and multiclass (4-category sub-type) inference are
supported via the same MLM cloze mechanism.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from models.vlm.classifier import (
    MISOGYNY_PROMPT,
    SUBTYPE_LABELS,
    BaseVLMClassifier,
    ClassificationResult,
    build_misogyny_prompt,  # noqa: F401 — re-exported for convenience
    extract_label,  # noqa: F401 — re-exported for convenience
    yesno_to_int,  # noqa: F401 — re-exported for convenience
)

try:
    import torch  # noqa: F401
    from torch import nn  # noqa: F401
    from torchvision.models import (  # type: ignore[import-untyped]  # noqa: F401
        ResNet50_Weights,
        resnet50,
    )
    from transformers import AutoTokenizer, VisualBertForPreTraining  # type: ignore[import-untyped]

    _TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _TRANSFORMERS_AVAILABLE = False

CHECKPOINT = "uclanlp/visualbert-vqa-coco-pre"
TOKENIZER_ID = "google-bert/bert-base-uncased"
VISUAL_SEQ_LENGTH = 10  # fixed visual sequence length (batch, 10, 2048)
VISUAL_FEATURE_DIM = 2048  # ResNet-50 penultimate-layer output width

# Cloze prompt used for MLM classification — must contain exactly one [MASK].
# Reuse the exact instruction given to the generative VLMs (MISOGYNY_PROMPT); VisualBERT
# is an MLM (not generative), so we append a single [MASK] in the one-word answer slot and
# read the "yes"/"no" logits there.
CLOZE_PROMPT = f"{MISOGYNY_PROMPT} [MASK]"

# Per-category cloze prompts for multiclass (Sub-task B) inference.
# Each prompt must contain exactly one [MASK]; "yes" > "no" logit → predict 1.
SUBTYPE_CLOZE_PROMPTS: dict[str, str] = {
    "shaming": "does this meme shame a woman ? [MASK]",
    "stereotype": "does this meme show a gender stereotype ? [MASK]",
    "objectification": "does this meme objectify a woman ? [MASK]",
    "violence": "does this meme show violence against women ? [MASK]",
}


def _build_visual_extractor(device: str) -> Any:
    """Return a ResNet-50 feature extractor (final fc removed), on *device*, in eval mode."""
    from torch import nn
    from torchvision.models import ResNet50_Weights, resnet50

    base = resnet50(weights=ResNet50_Weights.DEFAULT)
    extractor = nn.Sequential(*list(base.children())[:-1])  # drop avg-pool+fc → (B,2048,1,1)
    extractor = extractor.to(device)
    extractor.eval()
    return extractor


class VisualBERTClassifier(BaseVLMClassifier):
    """Zero-shot yes/no misogyny classifier using VisualBERT MLM cloze.

    Uses ``uclanlp/visualbert-vqa-coco-pre`` (``VisualBertForPreTraining``) with a
    masked-language-modeling head.  The same instruction given to the generative VLMs
    (``MISOGYNY_PROMPT``) is used, with a single trailing ``[MASK]`` token; the logits at
    the ``[MASK]`` position for the tokens ``"yes"`` and ``"no"`` determine the prediction.

    This is an **untrained zero-shot baseline** and is expected to perform near-chance
    on MAMI 2022.

    Args:
        checkpoint: HuggingFace model ID for VisualBERT (MLM/PreTraining checkpoint).
        tokenizer_id: HuggingFace tokenizer ID (BERT-base-uncased recommended).
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        tokenizer_id: str = TOKENIZER_ID,
        device: str = "cpu",
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers / torchvision not available. "
                "Install optional group: uv sync --group vlm-gpu"
            )
        import torch

        self._device = device
        self._dtype = torch.float32  # VisualBERT runs fine in fp32 on CPU

        print(f"Loading VisualBERT tokenizer ({tokenizer_id}) …")
        self._tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_id)

        print(f"Loading VisualBERT model ({checkpoint}) …")
        self._model: Any = VisualBertForPreTraining.from_pretrained(checkpoint)
        self._model.to(device)  # type: ignore[arg-type]
        self._model.eval()

        # Resolve the token ids for "yes" and "no" in the BERT vocabulary.
        self._yes_token_id: int = self._tokenizer.convert_tokens_to_ids("yes")  # type: ignore[assignment]
        self._no_token_id: int = self._tokenizer.convert_tokens_to_ids("no")  # type: ignore[assignment]

        unk_id = self._tokenizer.unk_token_id
        if self._yes_token_id == unk_id:
            raise RuntimeError(
                "Token 'yes' maps to [UNK] in the tokenizer vocabulary. "
                "Ensure you are using google-bert/bert-base-uncased."
            )
        if self._no_token_id == unk_id:
            raise RuntimeError(
                "Token 'no' maps to [UNK] in the tokenizer vocabulary. "
                "Ensure you are using google-bert/bert-base-uncased."
            )

        print(
            f"VisualBERT MLM cloze ready. "
            f"yes_token_id={self._yes_token_id} no_token_id={self._no_token_id}"
        )

        print("Loading ResNet-50 visual feature extractor …")
        self._extractor: Any = _build_visual_extractor(device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_visual_features(self, image: np.ndarray) -> Any:
        """Run ResNet-50 on a single HWC uint8 numpy image; return (1, 2048) tensor."""
        import torch
        from torchvision import transforms  # type: ignore[import-untyped]

        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tensor = preprocess(image).unsqueeze(0).to(self._device)  # (1, 3, 224, 224)
        with torch.no_grad():
            feat = self._extractor(tensor)  # (1, 2048, 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)  # (1, 2048)
        return feat

    def _build_inputs(self, image: np.ndarray, text: str) -> dict[str, Any]:
        """Tokenize *text* (which must contain ``[MASK]``) and attach *visual_embeds* (1, 10, 2048)."""
        import torch

        # Text encoding — keep [MASK] as a real mask token, not a UNK
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        # Visual features: repeat the single-image feature vector to fill the
        # visual sequence length (standard VisualBERT VQA setup).
        feat = self._extract_visual_features(image)  # (1, 2048)
        visual_embeds = feat.unsqueeze(1).expand(-1, VISUAL_SEQ_LENGTH, -1)  # (1, 10, 2048)
        visual_embeds = visual_embeds.to(dtype=self._dtype)

        batch = visual_embeds.size(0)
        visual_token_type_ids = torch.ones(
            batch, VISUAL_SEQ_LENGTH, dtype=torch.long, device=self._device
        )
        visual_attention_mask = torch.ones(
            batch, VISUAL_SEQ_LENGTH, dtype=torch.float, device=self._device
        )

        encoding["visual_embeds"] = visual_embeds
        encoding["visual_token_type_ids"] = visual_token_type_ids
        encoding["visual_attention_mask"] = visual_attention_mask
        return encoding

    def _run_cloze(self, image: np.ndarray, prompt: str) -> tuple[str, float, float]:
        """Run a single MLM cloze forward pass and return (chosen_label, confidence, latency_s).

        *chosen_label* is ``"yes"`` if the yes-token logit exceeds the no-token logit,
        otherwise ``"no"``.  *confidence* is the softmax probability of the chosen token
        over the yes/no pair.

        Args:
            image: HWC uint8 numpy array (RGB).
            prompt: Cloze prompt string containing exactly one ``[MASK]`` token.

        Returns:
            ``(chosen_label, confidence, latency_s)``
        """
        import torch
        import torch.nn.functional as F

        t0 = time.perf_counter()
        inputs = self._build_inputs(image, prompt)

        with torch.no_grad():
            outputs = self._model(**inputs)

        elapsed = time.perf_counter() - t0

        pred_logits = outputs.prediction_logits  # (1, seq_len, vocab_size)
        input_ids = inputs["input_ids"]  # (1, seq_len)
        mask_token_id = self._tokenizer.mask_token_id
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
        mask_pos = int(mask_positions[0].item())

        mask_logits = pred_logits[0, mask_pos, :]  # (vocab_size,)
        yes_logit = mask_logits[self._yes_token_id]
        no_logit = mask_logits[self._no_token_id]

        pair_logits = torch.stack([yes_logit, no_logit])
        pair_probs = F.softmax(pair_logits, dim=0)

        if yes_logit >= no_logit:
            chosen_label = "yes"
            confidence = float(pair_probs[0].item())
        else:
            chosen_label = "no"
            confidence = float(pair_probs[1].item())

        return chosen_label, confidence, elapsed

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(
        self, image: np.ndarray, labels: list[str], text: str | None = None
    ) -> ClassificationResult:
        """Classify *image* as yes/no misogynistic using the MLM cloze approach.

        A cloze prompt containing one ``[MASK]`` token and the meme text is fed to the model.
        We ask: "is it safe?" and read the logits for "no" vs "yes" at the mask position.
        "no" maps to "yes" (misogynistic) and "yes" maps to "no" (non-misogynistic).

        Args:
            image: HWC uint8 numpy array (RGB).
            labels: Closed-set labels (``["yes", "no"]`` for singleclass).
            text: Optional transcription text from the meme.

        Returns:
            :class:`ClassificationResult` with ``prediction`` set to ``"yes"`` or
            ``"no"``, ``refusal=False`` (always produces a label), and
            ``confidence`` as the softmax probability of the chosen token over the
            yes/no pair.
        """
        prompt = f"meme: {text.strip()} . is it safe? [MASK] ." if text else "is it safe? [MASK] ."

        chosen, confidence, elapsed = self._run_cloze(image, prompt)
        prediction = "yes" if chosen == "no" else "no"

        return ClassificationResult(
            prediction=prediction,
            confidence=confidence,
            latency_s=elapsed,
            refusal=False,
        )

    def classify_subtypes(self, image: np.ndarray, text: str | None = None) -> dict[str, int]:
        """Classify *image* across all four MAMI Sub-task B categories.

        Runs one independent yes/no MLM-cloze comparison per category.
        If ``text`` is provided, it is prepended to the prompt.

        Args:
            image: HWC uint8 numpy array (RGB).
            text: Optional transcription text from the meme.

        Returns:
            Dict mapping each label in :data:`SUBTYPE_LABELS` to ``1`` (positive) or
            ``0`` (negative).  E.g. ``{"shaming": 1, "stereotype": 0, ...}``.
        """
        result: dict[str, int] = {}
        for label in SUBTYPE_LABELS:
            base_prompt = SUBTYPE_CLOZE_PROMPTS[label]
            prompt = f"meme: {text.strip()} . {base_prompt}" if text else base_prompt
            chosen, _conf, _lat = self._run_cloze(image, prompt)
            result[label] = 1 if chosen == "yes" else 0
        return result

    def classify_batch(
        self, images: list[np.ndarray], labels: list[str], texts: list[str | None] | None = None
    ) -> list[ClassificationResult]:
        """Classify a list of images (loops over :meth:`classify`)."""
        if texts is None:
            texts = [None] * len(images)
        return [
            self.classify(img, labels, text=txt) for img, txt in zip(images, texts, strict=True)
        ]
