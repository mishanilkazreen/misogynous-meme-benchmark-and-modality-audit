"""
Property-based tests for YOLO classification pipeline.

Tests classification output format, training data format, preprocessing,
visibility-stratified evaluation, and inference performance.
"""

import time

from hypothesis import given, settings
from hypothesis import strategies as st
import torch

from models.yolo import (
    ClassificationResult,
    EvaluationMetrics,
    YOLOClassifier,
    YOLOEvaluator,
    YOLOTrainer,
    YOLOTrainingConfig,
)


class TestYOLOOutputFormat:
    """
    Property: YOLO classification output format.

    For any image processed by YOLO, the output should contain
    classification results with confidence scores in [0, 1].
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=2),
        height=st.sampled_from([128, 256, 416]),
        width=st.sampled_from([128, 256, 416]),
    )
    @settings(max_examples=50, deadline=None)
    def test_output_contains_valid_logits(self, batch_size, height, width):
        """For any image, YOLO output should have valid logit format."""
        model = YOLOClassifier(num_classes=10)
        image = torch.randn(batch_size, 3, height, width)

        outputs = model(image)

        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (batch_size, 10)

    @given(
        height=st.sampled_from([128, 256, 416]),
        width=st.sampled_from([128, 256, 416]),
    )
    @settings(max_examples=50, deadline=None)
    def test_classification_has_valid_confidence(self, height, width):
        """For any classification, confidence should be in [0, 1]."""
        model = YOLOClassifier(num_classes=10)
        image = torch.randn(1, 3, height, width)

        result = model.predict(image)

        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.confidence <= 1.0

    @given(
        height=st.sampled_from([128, 256, 416]),
        width=st.sampled_from([128, 256, 416]),
    )
    @settings(max_examples=50, deadline=None)
    def test_classification_has_valid_class(self, height, width):
        """For any classification, predicted class should be valid."""
        model = YOLOClassifier(num_classes=10)
        image = torch.randn(1, 3, height, width)

        result = model.predict(image)

        assert 0 <= result.predicted_class < 10

    @given(
        height=st.sampled_from([128, 256, 416]),
        width=st.sampled_from([128, 256, 416]),
    )
    @settings(max_examples=50, deadline=None)
    def test_classification_has_visibility_level(self, height, width):
        """For any classification, visibility_level should be 'high' or 'low'."""
        model = YOLOClassifier(num_classes=10)
        image = torch.randn(1, 3, height, width)

        result = model.predict(image)

        assert result.visibility_level in ["high", "low"]

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.sampled_from([128, 256]),
        width=st.sampled_from([128, 256]),
    )
    @settings(max_examples=30, deadline=None)
    def test_batch_prediction_returns_list(self, batch_size, height, width):
        """For batch prediction, should return list of results per image."""
        model = YOLOClassifier(num_classes=10)
        images = torch.randn(batch_size, 3, height, width)

        results = model.predict_batch(images)

        assert isinstance(results, list)
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, ClassificationResult)


class TestYOLOTrainingDataFormat:
    """
    Property: YOLO training data format.

    For any YOLO training batch, labels should be valid class indices.
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        num_classes=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_training_accepts_valid_labels(self, batch_size, num_classes):
        """Training should accept valid classification labels."""
        model = YOLOClassifier(num_classes=num_classes)
        config = YOLOTrainingConfig(batch_size=batch_size, epochs=1)
        trainer = YOLOTrainer(model, config)

        images = torch.randn(batch_size, 3, 128, 128)
        labels = torch.randint(0, num_classes, (batch_size,))

        # Should not raise
        logits = model(images.to(trainer.device))
        loss = trainer.criterion(logits, labels.to(trainer.device))

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    @given(batch_size=st.integers(min_value=1, max_value=4))
    @settings(max_examples=50, deadline=None)
    def test_cross_entropy_loss_computes(self, batch_size):
        """Cross entropy loss should compute for valid inputs."""
        model = YOLOClassifier(num_classes=10)
        config = YOLOTrainingConfig(batch_size=batch_size, epochs=1)
        trainer = YOLOTrainer(model, config)

        logits = torch.randn(batch_size, 10)
        labels = torch.randint(0, 10, (batch_size,))

        loss = trainer.criterion(logits, labels)

        assert loss.item() > 0


class TestYOLOPreprocessingConfiguration:
    """
    Property: Preprocessing configuration support.

    For any YOLO training with preprocessing enabled, blur and equalization
    should be applied to training images.
    """

    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
    )
    @settings(max_examples=50, deadline=None)
    def test_preprocessing_enabled_applies_transformations(self, height, width):
        """When preprocessing is enabled, blur and equalization should be applied."""
        model = YOLOClassifier(num_classes=10)
        config = YOLOTrainingConfig(preprocessing=True)
        trainer = YOLOTrainer(model, config)

        assert trainer.preprocessor is not None
        assert trainer.preprocessor.apply_blur is True
        assert trainer.preprocessor.apply_equalization is True

        image = torch.rand(1, 3, height, width)
        preprocessed = trainer.preprocess_batch(image)

        assert preprocessed.shape == image.shape

    @given(
        height=st.sampled_from([64, 128, 256]),
        width=st.sampled_from([64, 128, 256]),
    )
    @settings(max_examples=50, deadline=None)
    def test_preprocessing_disabled_preserves_images(self, height, width):
        """When preprocessing is disabled, images should pass through unchanged."""
        model = YOLOClassifier(num_classes=10)
        config = YOLOTrainingConfig(preprocessing=False)
        trainer = YOLOTrainer(model, config)

        assert trainer.preprocessor is None

        image = torch.rand(1, 3, height, width)
        result = trainer.preprocess_batch(image)

        assert torch.allclose(result, image.to(trainer.device))

    @given(preprocessing=st.booleans())
    @settings(max_examples=20, deadline=None)
    def test_config_reflects_preprocessing_setting(self, preprocessing):
        """Trainer config should correctly reflect preprocessing setting."""
        model = YOLOClassifier(num_classes=10)
        config = YOLOTrainingConfig(preprocessing=preprocessing)
        trainer = YOLOTrainer(model, config)

        if preprocessing:
            assert trainer.preprocessor is not None
        else:
            assert trainer.preprocessor is None


class TestYOLOVisibilityStratifiedEvaluation:
    """
    Property: Visibility-stratified evaluation.

    For any evaluation run, the system should report separate accuracy metrics
    for high visibility and low visibility content.
    """

    @given(
        num_high_vis=st.integers(min_value=1, max_value=5),
        num_low_vis=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_stratified_metrics_computed(self, num_high_vis, num_low_vis):
        """Evaluation should compute separate metrics for high and low visibility."""
        model = YOLOClassifier(num_classes=10)
        evaluator = YOLOEvaluator(model)

        results = []
        labels = []
        visibility_levels = []

        for _ in range(num_high_vis):
            results.append(
                ClassificationResult(
                    is_hateful=True, confidence=0.8, predicted_class=5, visibility_level="high"
                )
            )
            labels.append(True)
            visibility_levels.append("high")

        for _ in range(num_low_vis):
            results.append(
                ClassificationResult(
                    is_hateful=True, confidence=0.6, predicted_class=3, visibility_level="low"
                )
            )
            labels.append(True)
            visibility_levels.append("low")

        metrics = evaluator.evaluate_from_results(results, labels, visibility_levels)

        assert 0.0 <= metrics.accuracy_high_vis <= 1.0
        assert 0.0 <= metrics.accuracy_low_vis <= 1.0

    def test_evaluation_metrics_structure(self):
        """EvaluationMetrics should have all required fields."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.8,
            recall=0.9,
            f1=0.85,
            accuracy_high_vis=0.9,
            accuracy_low_vis=0.8,
            total_samples=100,
            true_positives=80,
            false_positives=10,
            true_negatives=5,
            false_negatives=5,
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.8
        assert metrics.recall == 0.9
        assert metrics.f1 == 0.85


class TestYOLOInferencePerformance:
    """
    Property: YOLO inference performance.

    For any image, YOLO classification should complete in reasonable time.
    """

    @given(
        height=st.sampled_from([128, 256, 416]),
        width=st.sampled_from([128, 256, 416]),
    )
    @settings(max_examples=30, deadline=None)
    def test_inference_completes_for_various_sizes(self, height, width):
        """Inference should complete for various image sizes."""
        model = YOLOClassifier(num_classes=10)
        image = torch.randn(1, 3, height, width)

        start = time.perf_counter()
        _ = model.predict(image)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"Inference took {elapsed:.2f}s"

    @given(batch_size=st.integers(min_value=1, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_batch_inference_works(self, batch_size):
        """Batch inference should work for various batch sizes."""
        model = YOLOClassifier(num_classes=10)
        images = torch.randn(batch_size, 3, 256, 256)

        start = time.perf_counter()
        results = model.predict_batch(images)
        elapsed = time.perf_counter() - start

        assert len(results) == batch_size
        assert elapsed < 30.0

    def test_model_config_roundtrip(self):
        """Model config should survive save/load roundtrip."""
        model = YOLOClassifier(num_classes=10, conf_threshold=0.6)
        config = model.get_config()

        restored = YOLOClassifier.from_config(config)

        assert restored.num_classes == model.num_classes
        assert restored.conf_threshold == model.conf_threshold
