import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from PIL import Image
import torch

from autobenchmark.vlm_bench import (
    get_device,
    parse_vlm_prediction,
    run_vlm_benchmark,
    run_vlm_inference,
)


class TestVLMBench(unittest.TestCase):
    def test_get_device(self):
        dev = get_device("auto")
        self.assertIn(dev.type, ["cuda", "cpu"])

        dev_cpu = get_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

    def test_parse_vlm_prediction(self):
        target_classes = ["red", "blue", "green", "dark red"]

        # Word boundaries
        self.assertEqual(parse_vlm_prediction("The color is red.", target_classes), "red")

        # Case insensitive
        self.assertEqual(parse_vlm_prediction("It seems BLUE.", target_classes), "blue")

        # Longer class name first
        self.assertEqual(parse_vlm_prediction("This is dark red color", target_classes), "dark red")

        # Substring matching fallback
        self.assertEqual(parse_vlm_prediction("reddish", target_classes), "red")

        # No match
        self.assertIsNone(parse_vlm_prediction("yellow color here", target_classes))

    @patch("torch.cuda.is_available", return_value=False)
    def test_run_vlm_inference_florence(self, mock_cuda):
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.tensor([[[[1.0]]]]),
        }
        mock_processor.batch_decode.return_value = ["red"]

        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(dtype=torch.float32)])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

        image = Image.new("RGB", (10, 10))
        device = torch.device("cpu")

        result = run_vlm_inference(
            model=mock_model,
            processor=mock_processor,
            model_type="florence",
            image=image,
            prompt="Classify this image",
            device=device,
        )

        self.assertEqual(result, "red")
        mock_processor.assert_called_once_with(
            text="Classify this image", images=image, return_tensors="pt"
        )
        mock_model.generate.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_run_vlm_inference_qwen2_vl(self, mock_cuda):
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "templated prompt"
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.tensor([[[[1.0]]]]),
        }
        mock_processor.batch_decode.return_value = ["blue"]

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(dtype=torch.float32)])
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

        image = Image.new("RGB", (10, 10))
        device = torch.device("cpu")

        result = run_vlm_inference(
            model=mock_model,
            processor=mock_processor,
            model_type="qwen2_vl",
            image=image,
            prompt="Is it blue?",
            device=device,
        )

        self.assertEqual(result, "blue")
        mock_processor.apply_chat_template.assert_called_once()
        mock_processor.assert_called_once_with(
            text=["templated prompt"], images=[image], padding=True, return_tensors="pt"
        )
        mock_model.generate.assert_called_once()

    @patch("autobenchmark.vlm_bench.load_vlm")
    @patch("autobenchmark.vlm_bench.run_vlm_inference")
    def test_run_vlm_benchmark_pipeline(self, mock_inference, mock_load):
        # Set up mocks
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_inference.side_effect = ["This color is red.", "I see green here."]

        # Temp directories
        temp_dir = tempfile.mkdtemp()
        try:
            # Create annotations CSV
            csv_path = os.path.join(temp_dir, "labels.csv")
            df = pd.DataFrame({"filename": ["img1.png", "img2.png"], "label": ["red", "green"]})
            df.to_csv(csv_path, index=False)

            # Create dummy images
            Image.new("RGB", (10, 10)).save(os.path.join(temp_dir, "img1.png"))
            Image.new("RGB", (10, 10)).save(os.path.join(temp_dir, "img2.png"))

            data_cfg = {
                "dataset": {
                    "data_type": "vlm",
                    "dataset_dir": temp_dir,
                    "annotation_file": csv_path,
                    "image_col": "filename",
                    "label_col": "label",
                },
                "prompt": "Identify the color",
                "classes": ["red", "blue", "green"],
                "training": {"device": "cpu"},
            }

            model_cfg = {
                "config_name": "test_vlm_run",
                "models_to_run": [
                    {"name": "MockModel", "model_id": "mock/model-id", "type": "florence"}
                ],
                "optimize_metric": "Accuracy",
            }

            output_dir = os.path.join(temp_dir, "results")

            df_eval = run_vlm_benchmark(
                data_cfg=data_cfg, model_cfg=model_cfg, output_dir=output_dir
            )

            self.assertFalse(df_eval.empty)
            self.assertEqual(df_eval.iloc[0]["Model"], "MockModel")
            self.assertEqual(df_eval.iloc[0]["Accuracy"], 1.0)

            # Check files created
            eval_file = os.path.join(output_dir, "test_vlm_run_evaluation.csv")
            preds_file = os.path.join(output_dir, "test_vlm_run_predictions.csv")
            chart_file = os.path.join(output_dir, "test_vlm_run_comparison_bar.png")

            self.assertTrue(os.path.exists(eval_file))
            self.assertTrue(os.path.exists(preds_file))
            self.assertTrue(os.path.exists(chart_file))

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
