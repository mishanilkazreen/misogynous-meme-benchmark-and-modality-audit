"""EXP-06: prompt sensitivity of zero-shot Gemini on the MAMI 2022 test split.

Runs ``gemini-3.1-flash-lite`` (temperature 0, all safety categories ``BLOCK_NONE``, exactly as
``scripts/benchmark_gemini.py``) over the full 1000-meme MAMI test split with several prompt
variants, caches every raw API response, and scores the variants with the official SemEval-2022
Task 5 scorer (``scripts/official_mami_evaluation.py``).

Task A (binary) variants: P0 (paper prompt), P1 (paraphrased definition), P2 (no definition),
P3 (MAMI-taxonomy prompt). Task B (sub-types) variants: B0 (paper JSON prompt), B1 (four
independent yes/no questions in one JSON). Task B uses a task-specific system instruction because
the paper's shared instruction demands a one-word answer, which contradicts a JSON reply.

Refusals, safety blocks, unparsable replies and API errors are stored as distinct ``status``
values and are never converted into labels in the raw files. Raw responses are appended to
``results/revision/exp06_prompts/raw_<variant>.jsonl`` so an interrupted run resumes.

Usage:
    uv run python scripts/revision/prompt_sensitivity.py run --variants all --workers 16
    uv run python scripts/revision/prompt_sensitivity.py evaluate

Both commands exit non-zero if any variant is incomplete or any call failed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import io
from itertools import combinations
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any

# utils.dataset imports `datasets`, which must be imported before torch (OpenMP segfault).
from utils.dataset import DatasetManager  # isort: split

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import numpy as np
from PIL import Image
from sklearn.metrics import cohen_kappa_score

from scripts.official_mami_evaluation import compute_f1, compute_scoreA, compute_scoreB

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "revision" / "exp06_prompts"
COMMITTED_P0 = ROOT / "results" / "test" / "gemini_test.json"

MODEL_ID = "gemini-3.1-flash-lite"
SPLIT = "test"
TEMPERATURE = 0.0
SUBTYPES: tuple[str, ...] = ("shaming", "stereotype", "objectification", "violence")
PRICE_INPUT_PER_M = 0.25  # USD per 1M input tokens (Google AI Studio list price, paid tier)
PRICE_OUTPUT_PER_M = 1.50  # USD per 1M output tokens (thinking tokens billed as output)
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260708
MAX_ATTEMPTS = 8
MAX_API_ERRORS = 25
RETRYABLE_CODES = {429, 500, 502, 503, 504}
BLOCK_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}
REFUSAL_PATTERN = re.compile(
    r"\b(i can(?:no|')t|i cannot|i am unable|i'm unable|i won't|i will not|sorry)\b", re.I
)

# Exact text of `_SYSTEM_INSTRUCTION` in scripts/benchmark_gemini.py (used for the main benchmark).
SYSTEM_A = (
    "You are a content moderation research assistant analysing internet memes for an academic "
    "misogyny detection study. Your role is purely classificatory: answer with exactly one word "
    "— 'yes' or 'no'. Do not add explanations, warnings, or refusals."
)
# Task-specific instruction for Task B (fixes the one-word vs JSON contradiction).
SYSTEM_B = (
    "You are a content moderation research assistant analysing internet memes for an academic "
    "misogyny detection study. Your role is purely classificatory: reply with only the JSON "
    "object requested in the prompt. Do not add explanations, warnings, or refusals."
)

_SUBTYPE_DEFS = (
    "- shaming: insults or attacks a woman's body, appearance, or behaviour.\n"
    "- stereotype: portrays women through oversimplified gender roles.\n"
    "- objectification: reduces a woman to a sexual object.\n"
    "- violence: depicts or threatens physical or sexual violence against women.\n"
)


@dataclass(frozen=True)
class Variant:
    """One prompt configuration."""

    prompt_id: str
    task: str  # "A" or "B"
    system_id: str
    system: str
    prompt: str
    description: str


VARIANTS: dict[str, Variant] = {
    "P0": Variant(
        "P0",
        "A",
        "SYSTEM_A",
        SYSTEM_A,
        # Exact text of MISOGYNY_PROMPT in models/vlm/classifier.py (paper Task A prompt).
        "Is this meme misogynistic? A misogynistic meme demeans, objectifies, or expresses "
        "hostility toward women. Answer with exactly one word: yes or no.",
        "Paper Task A prompt (baseline)",
    ),
    "P1": Variant(
        "P1",
        "A",
        "SYSTEM_A",
        SYSTEM_A,
        "Does this meme express misogyny? We call a meme misogynistic when it degrades women, "
        "treats them as sexual objects, or conveys hatred or aggression towards them. Answer "
        "with exactly one word: yes or no.",
        "Paraphrased definition",
    ),
    "P2": Variant(
        "P2",
        "A",
        "SYSTEM_A",
        SYSTEM_A,
        "Is this meme misogynistic? Answer yes or no.",
        "Definition removed",
    ),
    "P3": Variant(
        "P3",
        "A",
        "SYSTEM_A",
        SYSTEM_A,
        "This is a meme. Following the MAMI task, a meme is misogynistic if it expresses at "
        "least one of these four types of misogyny:\n" + _SUBTYPE_DEFS + "Is this meme "
        "misogynistic? Answer with exactly one word: yes or no.",
        "MAMI taxonomy (4 sub-types) then yes/no",
    ),
    "B0": Variant(
        "B0",
        "B",
        "SYSTEM_B",
        SYSTEM_B,
        # Exact text of build_subtype_prompt() in models/vlm/classifier.py (paper Task B).
        "This is a meme. Following the MAMI task, identify which TYPE(s) of misogyny it "
        "expresses, if any. A meme can express more than one type at the same time. Consider "
        "these four categories:\n" + _SUBTYPE_DEFS + "Reply with EXACTLY this JSON structure "
        'and nothing else:\n{"shaming": <true|false>, "stereotype": <true|false>, '
        '"objectification": <true|false>, "violence": <true|false>}',
        "Paper Task B JSON prompt (baseline)",
    ),
    "B1": Variant(
        "B1",
        "B",
        "SYSTEM_B",
        SYSTEM_B,
        "This is a meme. Answer the four questions below independently; each is a separate "
        "yes/no decision.\n"
        "shaming: Does this meme shame or insult a woman's body, appearance, or behaviour "
        "(for example body-shaming or slut-shaming)?\n"
        "stereotype: Does this meme reinforce oversimplified or traditional gender stereotypes "
        "about women?\n"
        "objectification: Does this meme reduce a woman to a sexual object or focus on her "
        "body for sexual purposes?\n"
        "violence: Does this meme depict, encourage, or threaten physical or sexual violence "
        "against women?\n"
        "Reply with EXACTLY this JSON structure and nothing else:\n"
        '{"shaming": "<yes|no>", "stereotype": "<yes|no>", "objectification": "<yes|no>", '
        '"violence": "<yes|no>"}',
        "Four per-sub-type yes/no questions in one JSON",
    ),
}


class QuotaExhaustedError(RuntimeError):
    """Raised when rate-limit retries are exhausted (quota or budget failure)."""


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


def _tensor_to_png(image: Any) -> bytes:
    """Encode a dataset image exactly as benchmark_gemini.py does (CHW float -> HWC uint8 PNG)."""
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def load_labels() -> dict[str, dict[str, int]]:
    """Return gold labels for the test split keyed by image_id (without loading pixels twice)."""
    dataset = DatasetManager().load_dataset(split=SPLIT)
    labels: dict[str, dict[str, int]] = {}
    for index in range(len(dataset)):
        row = dataset[index]
        labels[str(row["image_id"])] = {k: int(row[k]) for k in ("misogynous", *SUBTYPES)}
    return labels


def load_images(ids_needed: set) -> dict[str, bytes]:
    """Return PNG bytes for the requested test image ids."""
    dataset = DatasetManager().load_dataset(split=SPLIT)
    images: dict[str, bytes] = {}
    for index in range(len(dataset)):
        row = dataset[index]
        image_id = str(row["image_id"])
        if image_id in ids_needed:
            images[image_id] = _tensor_to_png(row["image"])
    return images


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------


def parse_task_a(text: str) -> int | None:
    """Return 1/0 if the reply contains exactly one of yes/no, else None (unparsable)."""
    found = set(re.findall(r"\b(yes|no)\b", text.lower()))
    if len(found) != 1:
        return None
    return 1 if found == {"yes"} else 0


def _to_binary(value: Any, allow_yes_no: bool) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if allow_yes_no and isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
        return 1 if value.strip().lower() == "yes" else 0
    return None


def parse_task_b(text: str, allow_yes_no: bool) -> dict[str, int] | None:
    """Parse a JSON reply with all four sub-type keys; return None if anything is missing."""
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return None
    candidate = match.group(0)
    obj: Any = None
    for attempt in (candidate, candidate.replace("'", '"')):
        try:
            obj = json.loads(attempt)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(obj, dict):
        return None
    parsed: dict[str, int] = {}
    for key in SUBTYPES:
        value = _to_binary(obj.get(key), allow_yes_no)
        if value is None:
            return None
        parsed[key] = value
    return parsed


# --------------------------------------------------------------------------------------------
# API calls
# --------------------------------------------------------------------------------------------


def _safety_settings() -> list[genai_types.SafetySetting]:
    """BLOCK_NONE on the four categories used by scripts/benchmark_gemini.py."""
    categories = (
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
    return [
        genai_types.SafetySetting(category=c, threshold=genai_types.HarmBlockThreshold.BLOCK_NONE)
        for c in categories
    ]


def _classify_reply(variant: Variant, response: Any) -> dict[str, Any]:
    """Turn an API response into status/raw_response/parsed fields."""
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None) if feedback else None
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    finish_name = getattr(finish, "name", None) or (str(finish) if finish else None)
    block_name = getattr(block_reason, "name", None) or (
        str(block_reason) if block_reason else None
    )
    text = ""
    if candidates and getattr(candidates[0], "content", None) is not None:
        parts = candidates[0].content.parts or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not p.thought)
    text = text.strip()
    usage = getattr(response, "usage_metadata", None)
    fields: dict[str, Any] = {
        "raw_response": text,
        "finish_reason": finish_name,
        "block_reason": block_name,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", None) or 0,
            "thought_tokens": getattr(usage, "thoughts_token_count", None) or 0,
        },
        "parsed": None,
    }
    if block_name or (finish_name in BLOCK_FINISH_REASONS):
        fields["status"] = "blocked"
        return fields
    if not text:
        fields["status"] = "refusal"
        return fields
    parsed: Any = (
        parse_task_a(text)
        if variant.task == "A"
        else parse_task_b(text, allow_yes_no=variant.prompt_id == "B1")
    )
    if parsed is not None:
        fields["status"] = "ok"
        fields["parsed"] = parsed
    elif REFUSAL_PATTERN.search(text):
        fields["status"] = "refusal"
    else:
        fields["status"] = "unparsable"
    return fields


def call_gemini(client: genai.Client, variant: Variant, image_id: str, png: bytes) -> dict:
    """Query Gemini once with backoff; return a raw-cache record (never raises on API errors
    except when rate-limit retries are exhausted)."""
    config = genai_types.GenerateContentConfig(
        system_instruction=variant.system,
        safety_settings=_safety_settings(),
        temperature=TEMPERATURE,
    )
    image_part = genai_types.Part.from_bytes(data=png, mime_type="image/png")
    contents: list[genai_types.PartUnionDict] = [variant.prompt, image_part]
    base: dict[str, Any] = {
        "image_id": image_id,
        "prompt_id": variant.prompt_id,
        "task": variant.task,
        "system_id": variant.system_id,
        "model": MODEL_ID,
        "split": SPLIT,
        "temperature": TEMPERATURE,
    }
    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=MODEL_ID, contents=contents, config=config
            )
        except genai_errors.APIError as exc:
            last_error = f"APIError {exc.code}: {exc.message}"
            if exc.code not in RETRYABLE_CODES:
                break
            if exc.code == 429 and attempt == MAX_ATTEMPTS - 1:
                raise QuotaExhaustedError(last_error) from exc
        except (OSError, TimeoutError) as exc:  # network-level failures (httpx errors subclass)
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            record = {**base, **_classify_reply(variant, response), "error": None}
            record["latency_s"] = round(time.perf_counter() - start, 3)
            record["attempts"] = attempt + 1
            record["timestamp"] = datetime.now(timezone.utc).isoformat()
            return record
        time.sleep(min(2.0 * 2**attempt, 90.0) + random.uniform(0, 1.5))
    return {
        **base,
        "status": "api_error",
        "raw_response": None,
        "parsed": None,
        "error": last_error,
        "attempts": MAX_ATTEMPTS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------------------------
# Raw cache
# --------------------------------------------------------------------------------------------


def raw_path(prompt_id: str) -> Path:
    """Path of the raw cache file for a variant."""
    return OUT_DIR / f"raw_{prompt_id}.jsonl"


def load_raw(prompt_id: str) -> dict[str, dict]:
    """Latest record per image_id (later lines override earlier ones, e.g. api_error retries)."""
    path = raw_path(prompt_id)
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                records[str(rec["image_id"])] = rec
    return records


def run_variant(
    client: genai.Client,
    variant: Variant,
    images: dict[str, bytes],
    all_ids: list[str],
    workers: int,
    limit: int | None,
) -> tuple[int, int, bool]:
    """Run pending images for one variant. Returns (n_done_now, n_api_errors, quota_hit)."""
    existing = load_raw(variant.prompt_id)
    pending = [i for i in all_ids if existing.get(i, {}).get("status", "api_error") == "api_error"]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(f"[{variant.prompt_id}] complete ({len(existing)} cached)")
        return 0, 0, False
    print(f"[{variant.prompt_id}] {len(pending)} pending, {workers} workers")
    lock = threading.Lock()
    stop = threading.Event()
    counters = {"done": 0, "errors": 0}
    quota_hit = False
    started = time.time()

    def worker(image_id: str) -> None:
        if stop.is_set():
            return
        record = call_gemini(client, variant, image_id, images[image_id])
        with lock:
            with raw_path(variant.prompt_id).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            counters["done"] += 1
            if record["status"] == "api_error":
                counters["errors"] += 1
                print(f"  [{variant.prompt_id}] {image_id}: {record['error']}")
                if counters["errors"] > MAX_API_ERRORS:
                    stop.set()
            if counters["done"] % 100 == 0:
                print(f"  [{variant.prompt_id}] {counters['done']}/{len(pending)}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i) for i in pending]
        for fut in as_completed(futures):
            try:
                fut.result()
            except QuotaExhaustedError as exc:
                if not quota_hit:
                    print(f"  [{variant.prompt_id}] QUOTA/RATE LIMIT EXHAUSTED: {exc}")
                quota_hit = True
                stop.set()
    with (OUT_DIR / "run_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "prompt_id": variant.prompt_id,
                    "model": MODEL_ID,
                    "split": SPLIT,
                    "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                    "wall_s": round(time.time() - started, 1),
                    "requests_written": counters["done"],
                    "api_errors": counters["errors"],
                    "quota_hit": quota_hit,
                    "workers": workers,
                }
            )
            + "\n"
        )
    return counters["done"], counters["errors"], quota_hit


def cmd_run(args: argparse.Namespace) -> int:
    """Run the requested variants; non-zero exit on any failure or incompleteness."""
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set", file=sys.stderr)
        return 2
    ids = list(args.variants)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    all_ids = list(labels)
    needed: set = set()
    for pid in ids:
        existing = load_raw(pid)
        needed |= {
            i for i in all_ids if existing.get(i, {}).get("status", "api_error") == "api_error"
        }
    images = load_images(needed) if needed else {}
    print(f"Loaded {len(all_ids)} test ids; encoded {len(images)} images")
    client = genai.Client(api_key=api_key)
    failed = False
    for pid in ids:
        _, errors, quota_hit = run_variant(
            client, VARIANTS[pid], images, all_ids, args.workers, args.limit
        )
        if quota_hit:
            print("Stopping: quota/budget failure. Partial raw files kept.", file=sys.stderr)
            return 3
        failed = failed or errors > 0
    incomplete = [p for p in ids if _n_final(p, all_ids) < len(all_ids)]
    if incomplete:
        print(f"Incomplete variants: {incomplete}", file=sys.stderr)
    return 1 if failed or incomplete else 0


def _n_final(prompt_id: str, all_ids: list[str]) -> int:
    recs = load_raw(prompt_id)
    return sum(1 for i in all_ids if recs.get(i, {}).get("status", "api_error") != "api_error")


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def _truth_row(gold: dict[str, int]) -> list[bool]:
    return [bool(gold["misogynous"])] + [bool(gold[s]) for s in SUBTYPES]


def _subtype_f1_pos(pred: list[list[bool]], gold: list[list[bool]]) -> float:
    """Unweighted mean positive-class F1 over the four sub-types (paper's Subtype-F1)."""
    scores = []
    for j in range(1, 5):
        tp = sum(1 for p, g in zip(pred, gold, strict=True) if p[j] and g[j])
        fp = sum(1 for p, g in zip(pred, gold, strict=True) if p[j] and not g[j])
        fn = sum(1 for p, g in zip(pred, gold, strict=True) if not p[j] and g[j])
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(scores))


def _metrics_a(ids: list[str], preds: dict[str, int], labels: dict[str, dict[str, int]]) -> dict:
    truth = {i: _truth_row(labels[i]) for i in ids}
    sub = {i: [bool(preds[i]), False, False, False, False] for i in ids}
    acc = float(np.mean([preds[i] == labels[i]["misogynous"] for i in ids]))
    return {"n": len(ids), "acc": acc, "score_a": float(compute_scoreA(truth, sub))}


def _metrics_b(
    ids: list[str], preds: dict[str, dict[str, int]], labels: dict[str, dict[str, int]]
) -> dict:
    truth = {i: _truth_row(labels[i]) for i in ids}
    sub = {i: [any(preds[i].values())] + [bool(preds[i][s]) for s in SUBTYPES] for i in ids}
    gold_rows = [truth[i] for i in ids]
    pred_rows = [sub[i] for i in ids]
    per_label = {
        s: float(compute_f1([r[j] for r in pred_rows], [g[j] for g in gold_rows]))
        for j, s in enumerate(SUBTYPES, start=1)
    }
    return {
        "n": len(ids),
        "score_b": float(compute_scoreB(truth, sub)),
        "subtype_f1": _subtype_f1_pos(pred_rows, gold_rows),
        "em": float(np.mean([p[1:] == g[1:] for p, g in zip(pred_rows, gold_rows, strict=True)])),
        "per_label_macro_f1": per_label,
    }


def _bootstrap(ids: list[str], fn: Any, keys: list[str]) -> dict[str, list[float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: dict[str, list[float]] = {k: [] for k in keys}
    arr = np.array(ids)
    for _ in range(N_BOOTSTRAP):
        sample = list(arr[rng.integers(0, len(arr), len(arr))])
        try:
            m = fn(sample)
        except ZeroDivisionError:
            continue
        for k in keys:
            draws[k].append(m[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()
    }


def _status_rates(recs: dict[str, dict], all_ids: list[str]) -> dict:
    counts = dict.fromkeys(("ok", "refusal", "blocked", "unparsable", "api_error", "missing"), 0)
    for i in all_ids:
        counts[recs[i]["status"] if i in recs else "missing"] += 1
    n = len(all_ids)
    return {
        "counts": counts,
        "refusal_rate": (counts["refusal"] + counts["blocked"]) / n,
        "blocked_rate": counts["blocked"] / n,
        "parse_failure_rate": counts["unparsable"] / n,
        "coverage": counts["ok"] / n,
    }


def _usage(recs: dict[str, dict]) -> dict:
    tin = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in recs.values())
    tout = sum(
        r.get("usage", {}).get("output_tokens", 0) + r.get("usage", {}).get("thought_tokens", 0)
        for r in recs.values()
    )
    return {
        "input_tokens": tin,
        "output_tokens": tout,
        "est_cost_usd": round(tin / 1e6 * PRICE_INPUT_PER_M + tout / 1e6 * PRICE_OUTPUT_PER_M, 4),
    }


def _committed_p0_agreement(recs: dict[str, dict]) -> dict | None:
    """Agreement of the P0 re-run with the committed paper run (default temperature)."""
    if not COMMITTED_P0.exists():
        return None
    rows = json.loads(COMMITTED_P0.read_text(encoding="utf-8"))[0]["sample_predictions"]
    old = {str(r["image_id"]): r["prediction"] for r in rows if r["prediction"] in (0, 1)}
    common = [i for i in old if recs.get(i, {}).get("status") == "ok"]
    if not common:
        return None
    a = [old[i] for i in common]
    b = [recs[i]["parsed"] for i in common]
    return {
        "n": len(common),
        "agreement": float(np.mean([x == y for x, y in zip(a, b, strict=True)])),
        "kappa": float(cohen_kappa_score(a, b)),
        "note": "committed run used default temperature; this re-run uses temperature 0",
    }


def cmd_evaluate(_: argparse.Namespace) -> int:
    """Compute metrics.json, prompts.md and summary.md from the raw caches."""
    labels = load_labels()
    all_ids = list(labels)
    out: dict[str, Any] = {
        "experiment": "EXP-06 prompt sensitivity",
        "model": MODEL_ID,
        "split": SPLIT,
        "n_test": len(all_ids),
        "temperature": TEMPERATURE,
        "safety": "BLOCK_NONE on harassment, hate speech, sexually explicit, dangerous content",
        "scorer": "scripts/official_mami_evaluation.py (compute_scoreA / compute_scoreB)",
        "primary_scoring": "answered subset (status == ok); failures are not labels",
        "secondary_scoring": "all 1000 memes; non-ok items scored as negative (no / all false)",
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "ci": "percentile 95%"},
        "pricing_usd_per_m": {"input": PRICE_INPUT_PER_M, "output": PRICE_OUTPUT_PER_M},
        "generated": datetime.now(timezone.utc).isoformat(),
        "variants": {},
    }
    incomplete: list[str] = []
    preds: dict[str, dict[str, Any]] = {}
    for pid, var in VARIANTS.items():
        recs = load_raw(pid)
        rates = _status_rates(recs, all_ids)
        entry: dict[str, Any] = {
            "task": var.task,
            "system_id": var.system_id,
            "description": var.description,
            **rates,
            "usage": _usage(recs),
        }
        if rates["counts"]["missing"] or rates["counts"]["api_error"]:
            incomplete.append(pid)
            entry["complete"] = False
        else:
            entry["complete"] = True
        ok_ids = [i for i in all_ids if recs.get(i, {}).get("status") == "ok"]
        p = {i: recs[i]["parsed"] for i in ok_ids}
        preds[pid] = p
        if ok_ids:
            if var.task == "A":
                full = {i: p.get(i, 0) for i in all_ids}
                entry["primary"] = _metrics_a(ok_ids, p, labels)
                entry["primary_ci95"] = _bootstrap(
                    ok_ids, lambda s, p=p: _metrics_a(s, p, labels), ["acc", "score_a"]
                )
                entry["secondary_all"] = _metrics_a(all_ids, full, labels)
            else:
                zero = dict.fromkeys(SUBTYPES, 0)
                fullb = {i: p.get(i, zero) for i in all_ids}
                entry["primary"] = _metrics_b(ok_ids, p, labels)
                entry["primary_ci95"] = _bootstrap(
                    ok_ids,
                    lambda s, p=p: _metrics_b(s, p, labels),
                    ["score_b", "subtype_f1", "em"],
                )
                entry["secondary_all"] = _metrics_b(all_ids, fullb, labels)
        if pid == "P0":
            entry["vs_committed_paper_run"] = _committed_p0_agreement(recs)
        out["variants"][pid] = entry

    out["agreement"] = _agreement(preds)
    out["spread"] = _spread(out["variants"])
    out["incomplete_variants"] = incomplete
    (OUT_DIR / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    write_prompts_md()
    write_summary_md(out)
    print(f"Wrote {OUT_DIR / 'metrics.json'}")
    if incomplete:
        print(f"Incomplete variants: {incomplete}", file=sys.stderr)
        return 1
    return 0


def _agreement(preds: dict[str, dict[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for task, group in (("A", ["P0", "P1", "P2", "P3"]), ("B", ["B0", "B1"])):
        for x, y in combinations(group, 2):
            common = [i for i in preds[x] if i in preds[y]]
            if not common:
                continue
            if task == "A":
                a = [preds[x][i] for i in common]
                b = [preds[y][i] for i in common]
                result[f"{x}-{y}"] = {
                    "n": len(common),
                    "agreement": float(np.mean([u == v for u, v in zip(a, b, strict=True)])),
                    "kappa": float(cohen_kappa_score(a, b)),
                }
            else:
                per = {
                    s: float(
                        cohen_kappa_score(
                            [preds[x][i][s] for i in common], [preds[y][i][s] for i in common]
                        )
                    )
                    for s in SUBTYPES
                }
                result[f"{x}-{y}"] = {
                    "n": len(common),
                    "kappa_per_subtype": per,
                    "kappa_mean": float(np.mean(list(per.values()))),
                    "exact_vector_agreement": float(
                        np.mean([preds[x][i] == preds[y][i] for i in common])
                    ),
                }
    return result


def _spread(variants: dict[str, Any]) -> dict:
    spread: dict[str, Any] = {}
    for task, keys in (("A", ["acc", "score_a"]), ("B", ["score_b", "subtype_f1", "em"])):
        group = {k: v for k, v in variants.items() if v["task"] == task and "primary" in v}
        for scoring in ("primary", "secondary_all"):
            for key in keys:
                vals = {pid: v[scoring][key] for pid, v in group.items()}
                if len(vals) >= 2:
                    spread[f"{task}.{scoring}.{key}"] = {
                        "max": max(vals.values()),
                        "min": min(vals.values()),
                        "max_minus_min": max(vals.values()) - min(vals.values()),
                        "argmax": max(vals, key=lambda k: vals[k]),
                        "argmin": min(vals, key=lambda k: vals[k]),
                    }
    return spread


# --------------------------------------------------------------------------------------------
# Markdown outputs
# --------------------------------------------------------------------------------------------


def write_prompts_md() -> None:
    """Write the exact system instructions and prompts sent to the API."""
    lines = [
        "# EXP-06 prompt variants (exact text sent to the API)",
        "",
        f"Model `{MODEL_ID}`, temperature {TEMPERATURE}, all four safety categories `BLOCK_NONE`.",
        "The image (PNG) is sent after the prompt text, as in `scripts/benchmark_gemini.py`.",
        "",
        "## System instructions",
        "",
        "`SYSTEM_A` (Task A; verbatim `_SYSTEM_INSTRUCTION` of `scripts/benchmark_gemini.py`):",
        "",
        "```text",
        SYSTEM_A,
        "```",
        "",
        "`SYSTEM_B` (Task B; task-specific, replaces the one-word instruction that",
        "contradicts the JSON reply format):",
        "",
        "```text",
        SYSTEM_B,
        "```",
        "",
        "## Prompts",
        "",
    ]
    for var in VARIANTS.values():
        lines += [
            f"### {var.prompt_id}: {var.description}",
            "",
            f"Task {var.task}, system instruction `{var.system_id}`.",
            "",
            "```text",
            var.prompt,
            "```",
            "",
        ]
    (OUT_DIR / "prompts.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _fmt_ci(entry: dict, key: str) -> str:
    lo, hi = entry["primary_ci95"][key]
    return f"{entry['primary'][key]:.4f} [{lo:.4f}, {hi:.4f}]"


def write_summary_md(out: dict[str, Any]) -> None:
    """Write a short (<= 40 lines) human-readable summary of metrics.json."""
    v = out["variants"]
    lines = [
        "# EXP-06 prompt sensitivity (Gemini, MAMI test, n=1000)",
        "",
        f"`{MODEL_ID}`, temperature 0, `BLOCK_NONE`; official MAMI scorer; 95% bootstrap CI",
        f"({N_BOOTSTRAP} resamples). Primary scores use answered memes only; `all` scores the",
        "full 1000 with failures as negative. Refusal = refusal + safety block.",
        "",
        "| Var | Acc [CI] | Score A [CI] | Score A (all) | Refusal | Parse fail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pid in ("P0", "P1", "P2", "P3"):
        e = v[pid]
        if "primary" not in e:
            lines.append(f"| {pid} | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {pid} | {_fmt_ci(e, 'acc')} | {_fmt_ci(e, 'score_a')} | "
            f"{e['secondary_all']['score_a']:.4f} | {e['refusal_rate']:.1%} | "
            f"{e['parse_failure_rate']:.1%} |"
        )
    lines += [
        "",
        "| Var | Score B [CI] | Subtype-F1 [CI] | EM [CI] | Score B (all) | Refusal | Parse fail |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pid in ("B0", "B1"):
        e = v[pid]
        if "primary" not in e:
            lines.append(f"| {pid} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {pid} | {_fmt_ci(e, 'score_b')} | {_fmt_ci(e, 'subtype_f1')} | "
            f"{_fmt_ci(e, 'em')} | {e['secondary_all']['score_b']:.4f} | "
            f"{e['refusal_rate']:.1%} | {e['parse_failure_rate']:.1%} |"
        )
    sp = out["spread"]
    lines += ["", "Spread (max - min, primary):"]
    for key in ("A.primary.acc", "A.primary.score_a", "B.primary.score_b", "B.primary.em"):
        if key in sp:
            s = sp[key]
            lines.append(
                f"- {key}: {s['max_minus_min']:.4f} ({s['argmax']} {s['max']:.4f} vs "
                f"{s['argmin']} {s['min']:.4f})"
            )
    ag = out["agreement"]
    kap = [f"{k} {x['kappa']:.3f}" for k, x in ag.items() if "kappa" in x]
    lines += ["", "Cohen's kappa (Task A): " + ", ".join(kap)]
    if "B0-B1" in ag:
        lines.append(
            f"Cohen's kappa (Task B, mean over sub-types): {ag['B0-B1']['kappa_mean']:.3f}"
        )
    ref = v["P0"].get("vs_committed_paper_run")
    if ref:
        lines.append(
            f"P0 re-run vs committed paper run (default temperature): agreement "
            f"{ref['agreement']:.3f}, kappa {ref['kappa']:.3f} (n={ref['n']})."
        )
    cost = sum(e["usage"]["est_cost_usd"] for e in v.values())
    lines.append(f"Estimated API cost (list price): ${cost:.2f}.")
    if out["incomplete_variants"]:
        lines.append(f"INCOMPLETE variants: {', '.join(out['incomplete_variants'])}.")
    lines += ["", "Prompts: `prompts.md`. Full numbers: `metrics.json`."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _parse_variants(value: str) -> list[str]:
    if value == "all":
        return list(VARIANTS)
    ids = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in ids if v not in VARIANTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {unknown}")
    return ids


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="query Gemini (resumes from raw caches)")
    run.add_argument("--variants", type=_parse_variants, default=list(VARIANTS))
    run.add_argument("--workers", type=int, default=16)
    run.add_argument("--limit", type=int, default=None, help="max new requests per variant")
    sub.add_parser("evaluate", help="score cached responses")
    args = parser.parse_args()
    code = cmd_run(args) if args.cmd == "run" else cmd_evaluate(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
