"""
Vision-Language Model (VLM) benchmarking module for the autobenchmark package.
Loads open-source Hugging Face VLMs, performs zero-shot image classification,
computes evaluation metrics, and ranks models.
"""

import os
import re
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch


def get_device(device_setting="auto"):
    """Resolve torch device."""
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def load_vlm(model_id, model_type, device, quantization=None):
    """
    Load a pretrained VLM and its processor from Hugging Face.

    Args:
        model_id: Hugging Face model identifier (e.g., 'microsoft/Florence-2-base').
        model_type: 'florence' or 'qwen2_vl'.
        device: Torch device.
        quantization: '4bit', '8bit', or None.

    Returns:
        model, processor
    """
    from transformers import AutoProcessor

    load_kwargs = {}
    if quantization == "8bit":
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif quantization == "4bit":
        load_kwargs["load_in_4bit"] = True
        load_kwargs["device_map"] = "auto"
    # Default precision settings
    elif device.type == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    print(f"Loading VLM model '{model_id}' ({model_type}) on device: {device}...")

    if model_type == "florence":
        # Apply Hugging Face Florence-2 configuration dynamic module compatibility patch
        from transformers.configuration_utils import PretrainedConfig

        PretrainedConfig.forced_bos_token_id = None

        # Apply tokenizer base class compatibility patch (RobertaTokenizer attribute missing check)
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        PreTrainedTokenizerBase.additional_special_tokens = property(
            lambda self: self.special_tokens_map.get("additional_special_tokens", [])
        )

        # Apply Cache indexing monkey-patches for transformers v5 compatibility
        try:
            from transformers.cache_utils import Cache, EncoderDecoderCache

            def Cache_getitem(self, index):
                if index < 0 or index >= len(self.layers):
                    return None
                layer = self.layers[index]
                if layer.keys is None:
                    return None
                return (layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None))

            Cache.__getitem__ = Cache_getitem

            def EncoderDecoderCache_getitem(self, index):
                if index < 0 or index >= len(self.self_attention_cache):
                    return None
                self_layer = self.self_attention_cache.layers[index]
                cross_layer = self.cross_attention_cache.layers[index]
                if self_layer.keys is None:
                    return None
                self_attn = (
                    self_layer.keys,
                    self_layer.values,
                    getattr(self_layer, "_sliding_window_tensor", None),
                )
                cross_attn = (
                    cross_layer.keys,
                    cross_layer.values,
                    getattr(cross_layer, "_sliding_window_tensor", None),
                )
                return self_attn + cross_attn

            EncoderDecoderCache.__getitem__ = EncoderDecoderCache_getitem
        except Exception as cache_patch_err:
            print(f"  Warning: Failed to apply cache subscripting patch: {cache_patch_err}")

        # Pre-load dynamic class to populate sys.modules for dynamic class monkey-patching
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            get_class_from_dynamic_module(
                "modeling_florence2.Florence2ForConditionalGeneration", model_id
            )

            import sys

            modeling_module = None
            for name, module in sys.modules.items():
                if name.endswith("modeling_florence2"):
                    modeling_module = module
                    break
            if modeling_module is not None:
                cls = getattr(modeling_module, "Florence2PreTrainedModel", None)
                if cls is not None:
                    # Properties throw AttributeError during init when self.language_model isn't built yet
                    cls._supports_sdpa = property(
                        lambda self: self.language_model._supports_sdpa
                        if hasattr(self, "language_model")
                        else False
                    )
                    cls._supports_flash_attn_2 = property(
                        lambda self: self.language_model._supports_flash_attn_2
                        if hasattr(self, "language_model")
                        else False
                    )
        except Exception as patch_err:
            print(f"  Warning: Failed to apply dynamic model patch: {patch_err}")

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, **load_kwargs
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Patch prepare_inputs_for_generation on the loaded model class to safely handle Cache inputs
        try:
            if hasattr(model, "language_model"):
                language_model_class = model.language_model.__class__
                original_prepare = language_model_class.prepare_inputs_for_generation

                def patched_prepare(self, decoder_input_ids, past_key_values=None, **kwargs):
                    past_length = 0
                    if past_key_values is not None:
                        if hasattr(past_key_values, "get_seq_length"):
                            past_length = past_key_values.get_seq_length()
                        else:
                            try:
                                past_length = past_key_values[0][0].shape[2]
                            except Exception:
                                past_length = 0

                        if decoder_input_ids.shape[1] > past_length:
                            remove_prefix_length = past_length
                        else:
                            remove_prefix_length = decoder_input_ids.shape[1] - 1
                        decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

                    passed_past_key_values = past_key_values if past_length > 0 else None
                    kwargs["past_key_values"] = passed_past_key_values

                    model_inputs = original_prepare(
                        self, decoder_input_ids=decoder_input_ids, **kwargs
                    )
                    return model_inputs

                language_model_class.prepare_inputs_for_generation = patched_prepare
        except Exception as prep_patch_err:
            print(f"  Warning: Failed to patch prepare_inputs_for_generation: {prep_patch_err}")

        # Move model to device if not quantized (quantized models use device_map auto)
        if quantization not in ["4bit", "8bit"]:
            model = model.to(device)
    elif model_type == "qwen2_vl":
        # Qwen2VLForConditionalGeneration
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(model_id)
        if quantization not in ["4bit", "8bit"]:
            model = model.to(device)
    else:
        # Generic fallback
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(model_id)
        if quantization not in ["4bit", "8bit"]:
            model = model.to(device)

    return model, processor


def run_vlm_inference(model, processor, model_type, image, prompt, device):
    """Run zero-shot prompt prediction on an image."""
    try:
        if model_type == "florence":
            # Florence-2 model expects the prompt in the text input
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            # Florence-2 expects float16 if model is loaded in float16
            if next(model.parameters()).dtype == torch.float16 and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=3,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()

        elif model_type == "qwen2_vl":
            # Qwen2-VL chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            if next(model.parameters()).dtype == torch.float16 and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            generated_ids = model.generate(**inputs, max_new_tokens=64)
            # Trim the prompt tokens from generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=False)
            ]
            generated_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return generated_text.strip()

        else:
            # Generic vision-to-language fallback
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=64)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()

    except Exception as e:
        print(f"Error during VLM inference: {e}")
        return f"ERROR: {e}"


def parse_vlm_prediction(prediction_text, target_classes):
    """
    Parse the VLM raw text response and map it to one of the target classes
    using word boundaries and substring matching.
    """
    text_lower = prediction_text.lower()

    # Sort by length descending to match longer class names first
    sorted_classes = sorted(target_classes, key=len, reverse=True)

    # Check for exact word boundaries first
    for cls in sorted_classes:
        pattern = r"\b" + re.escape(cls.lower()) + r"\b"
        if re.search(pattern, text_lower):
            return cls

    # Substring matching fallback
    for cls in sorted_classes:
        if cls.lower() in text_lower:
            return cls

    return None


def run_vlm_benchmark(data_cfg, model_cfg, output_dir, init_config=None):
    """
    Run VLM zero-shot benchmarking: load datasets, run VLM predictions,
    parse responses, calculate metrics, and save rankings.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_cfg = data_cfg.get("dataset", {})
    dataset_dir = dataset_cfg.get("dataset_dir", "")
    annotation_file = dataset_cfg.get("annotation_file", "")

    # Resolve relative paths
    if init_config:
        base_dir = init_config.get("system", {}).get("base_dir", "")
        if base_dir:
            if not os.path.isabs(dataset_dir):
                dataset_dir = os.path.join(base_dir, dataset_dir)
            if not os.path.isabs(annotation_file):
                annotation_file = os.path.join(base_dir, annotation_file)

    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotations file not found at: {annotation_file}")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    # Load annotations
    df_ann = pd.read_csv(annotation_file)
    image_col = dataset_cfg.get("image_col", "filename")
    label_col = dataset_cfg.get("label_col", "label")

    if image_col not in df_ann.columns or label_col not in df_ann.columns:
        raise ValueError(f"Annotation file must contain columns '{image_col}' and '{label_col}'")

    prompt = data_cfg.get("prompt", "")
    target_classes = data_cfg.get("classes", [])
    if not target_classes:
        target_classes = sorted(df_ann[label_col].dropna().unique().tolist())

    device = get_device(data_cfg.get("training", {}).get("device", "auto"))
    models_list = model_cfg.get("models_to_run", [])
    model_config_name = model_cfg.get("config_name", "vlm_benchmark")
    optimize_metric = model_cfg.get("optimize_metric", "Accuracy")

    print(f"Starting VLM Benchmarking: {model_config_name}")
    print(f"Images count: {len(df_ann)}")
    print(f"Target classes: {target_classes}")

    # DataFrame to save all raw outputs and parsed predictions
    predictions_df = pd.DataFrame(
        {"filename": df_ann[image_col], "ground_truth": df_ann[label_col]}
    )

    evaluation_results = []

    for model_spec in models_list:
        model_name = model_spec["name"]
        model_id = model_spec["model_id"]
        model_type = model_spec.get("type", "generic")
        quantization = model_spec.get("quantization", None)

        print(f"\nBenchmarking Model: {model_name} ({model_id})")

        # Load VLM
        try:
            model, processor = load_vlm(model_id, model_type, device, quantization)
        except Exception as e:
            print(f"  ERROR: Failed to load VLM model {model_id}: {e}")
            evaluation_results.append(
                {
                    "Model": model_name,
                    "Model_ID": model_id,
                    "Status": f"LOAD_FAILED: {e}",
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F1": 0.0,
                    "Avg_Latency_Sec": 0.0,
                }
            )
            continue

        raw_preds = []
        parsed_labels = []
        latencies = []

        # Run inference
        for idx, row in df_ann.iterrows():
            img_name = row[image_col]
            img_path = os.path.join(dataset_dir, img_name)

            if not os.path.exists(img_path):
                print(f"  Warning: Image not found: {img_path}")
                raw_preds.append("IMAGE_NOT_FOUND")
                parsed_labels.append(None)
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  Warning: Failed to load image {img_path}: {e}")
                raw_preds.append(f"LOAD_ERROR: {e}")
                parsed_labels.append(None)
                continue

            # Run inference and time it
            start_t = time.time()
            raw_pred = run_vlm_inference(model, processor, model_type, image, prompt, device)
            latency = time.time() - start_t

            parsed = parse_vlm_prediction(raw_pred, target_classes)

            raw_preds.append(raw_pred)
            parsed_labels.append(parsed)
            latencies.append(latency)

            try:
                print(
                    f"  [{idx + 1}/{len(df_ann)}] {img_name} -> Raw: '{raw_pred}' | Parsed: '{parsed}' | Time: {latency:.2f}s"
                )
            except UnicodeEncodeError:
                safe_pred = raw_pred.encode("ascii", errors="replace").decode("ascii")
                print(
                    f"  [{idx + 1}/{len(df_ann)}] {img_name} -> Raw: '{safe_pred}' | Parsed: '{parsed}' | Time: {latency:.2f}s"
                )

        # Clean up memory
        del model
        del processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Log to predictions DataFrame
        predictions_df[f"{model_name}_raw"] = raw_preds
        predictions_df[f"{model_name}_parsed"] = parsed_labels

        # Calculate metrics
        ground_truth = df_ann[label_col].tolist()

        # Filter valid predictions (exclude None for calculation)
        valid_indices = [i for i, val in enumerate(parsed_labels) if val is not None]
        if not valid_indices:
            print(
                f"  WARNING: Model '{model_name}' produced no parsable predictions matching target classes."
            )
            evaluation_results.append(
                {
                    "Model": model_name,
                    "Model_ID": model_id,
                    "Status": "NO_PARSABLE_PREDICTIONS",
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F1": 0.0,
                    "Avg_Latency_Sec": np.mean(latencies) if latencies else 0.0,
                }
            )
            continue

        gt_filtered = [ground_truth[i] for i in valid_indices]
        pred_filtered = [parsed_labels[i] for i in valid_indices]

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # Accuracy over all images (unparsed counts as wrong)
        acc = accuracy_score(
            ground_truth, [val if val is not None else "UNPARSED" for val in parsed_labels]
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_filtered, pred_filtered, average="macro", zero_division=0
        )
        avg_latency = np.mean(latencies)

        print(
            f"  {model_name} Results: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Avg Latency={avg_latency:.2f}s"
        )

        evaluation_results.append(
            {
                "Model": model_name,
                "Model_ID": model_id,
                "Status": "OK",
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Avg_Latency_Sec": avg_latency,
            }
        )

    if not evaluation_results:
        print("No models were benchmarked successfully.")
        return pd.DataFrame()

    # Compile and rank results
    df_eval = pd.DataFrame(evaluation_results)

    # Sort descending by optimize_metric
    sort_col = optimize_metric
    if sort_col in df_eval.columns:
        df_eval = df_eval.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    # Save files
    eval_csv_path = os.path.join(output_dir, f"{model_config_name}_evaluation.csv")
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"\nSaved ranked VLM evaluations -> {eval_csv_path}")

    preds_csv_path = os.path.join(output_dir, f"{model_config_name}_predictions.csv")
    predictions_df.to_csv(preds_csv_path, index=False)
    print(f"Saved VLM predictions sheet -> {preds_csv_path}")

    # Generate bar chart comparison
    _generate_vlm_bar_chart(df_eval, optimize_metric, output_dir, model_config_name)

    return df_eval


def _generate_vlm_bar_chart(df_eval, metric_name, output_dir, config_name):
    """Plot horizontal bar chart ranking VLM performance on target metric."""
    successful = df_eval[df_eval["Status"] == "OK"].copy()
    if successful.empty:
        return

    if metric_name not in successful.columns:
        return

    successful = successful.sort_values(by=metric_name, ascending=True)

    _fig, ax = plt.subplots(figsize=(10, max(4, len(successful) * 0.6)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(successful)))

    bars = ax.barh(successful["Model"], successful[metric_name], color=colors)

    # Add value labels to bars
    for bar, val in zip(bars, successful[metric_name], strict=False):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_title(f"VLM Performance Ranking — {metric_name}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, min(1.0, successful[metric_name].max() * 1.15))
    plt.tight_layout()

    chart_path = os.path.join(output_dir, f"{config_name}_comparison_bar.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved performance chart -> {chart_path}")
