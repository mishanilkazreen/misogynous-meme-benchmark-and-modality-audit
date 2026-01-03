"""
Property-based tests for dataset composition.
Tests Property 2: Dataset composition completeness.
"""

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from utils.augmentation import BalancedSampler


class TestDatasetComposition:
    """
    Property 2: Dataset composition completeness.

    For any prepared training dataset, it should contain all four combinations:
    (high visibility + textual), (high visibility + symbolic),
    (low visibility + textual), (low visibility + symbolic).
    """

    @given(
        num_samples=st.integers(min_value=4, max_value=100),
    )
    @settings(max_examples=50)
    def test_balanced_sampler_returns_correct_count(self, num_samples):
        """BalancedSampler should return requested number of samples."""
        # Create annotations with all categories
        annotations = self._create_complete_annotations(20)

        sampler = BalancedSampler(annotations)
        indices = sampler.get_balanced_indices(num_samples)

        assert len(indices) == num_samples

    @given(
        samples_per_category=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_balanced_sampler_distributes_evenly(self, samples_per_category):
        """Samples should be roughly evenly distributed across categories."""
        # Create annotations with all categories
        annotations = self._create_complete_annotations(50)

        sampler = BalancedSampler(annotations)
        num_samples = samples_per_category * 4  # 4 categories
        indices = sampler.get_balanced_indices(num_samples)

        # Count samples per category
        category_counts = {
            "high_visibility_textual": 0,
            "high_visibility_symbolic": 0,
            "low_visibility_textual": 0,
            "low_visibility_symbolic": 0,
        }

        for idx in indices:
            ann = annotations[idx]
            key = f"{ann['visibility_level']}_visibility_{ann['message_type']}"
            if key in category_counts:
                category_counts[key] += 1

        # Each category should have roughly equal samples
        for count in category_counts.values():
            assert count >= samples_per_category - 1
            assert count <= samples_per_category + 1

    def test_composition_completeness_with_all_categories(self):
        """Dataset with all categories should report complete."""
        annotations = self._create_complete_annotations(20)

        sampler = BalancedSampler(annotations)
        completeness = sampler.check_composition_completeness()

        assert completeness["high_visibility_textual"] is True
        assert completeness["high_visibility_symbolic"] is True
        assert completeness["low_visibility_textual"] is True
        assert completeness["low_visibility_symbolic"] is True

    def test_composition_completeness_with_missing_category(self):
        """Dataset missing a category should report incomplete."""
        # Create annotations missing low_visibility_symbolic
        # Missing low_visibility_symbolic
        annotations = [
            {"visibility_level": "high", "message_type": "textual"},
            {"visibility_level": "high", "message_type": "symbolic"},
            {"visibility_level": "low", "message_type": "textual"},
        ]

        sampler = BalancedSampler(annotations)
        completeness = sampler.check_composition_completeness()

        assert completeness["high_visibility_textual"] is True
        assert completeness["high_visibility_symbolic"] is True
        assert completeness["low_visibility_textual"] is True
        assert completeness["low_visibility_symbolic"] is False

    def test_category_counts_accurate(self):
        """Category counts should match actual data."""
        annotations = [
            {"visibility_level": "high", "message_type": "textual"},
            {"visibility_level": "high", "message_type": "textual"},
            {"visibility_level": "low", "message_type": "symbolic"},
        ]

        sampler = BalancedSampler(annotations)
        counts = sampler.get_category_counts()

        assert counts["high_visibility_textual"] == 2
        assert counts["high_visibility_symbolic"] == 0
        assert counts["low_visibility_textual"] == 0
        assert counts["low_visibility_symbolic"] == 1

    @given(
        num_samples=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=20)
    def test_sampler_handles_imbalanced_data(self, num_samples):
        """Sampler should handle imbalanced categories gracefully."""
        # Create imbalanced annotations
        annotations = []
        # Many high_visibility_textual
        for _ in range(30):
            annotations.append({"visibility_level": "high", "message_type": "textual"})
        # Few of others
        annotations.append({"visibility_level": "high", "message_type": "symbolic"})
        annotations.append({"visibility_level": "low", "message_type": "textual"})
        annotations.append({"visibility_level": "low", "message_type": "symbolic"})

        sampler = BalancedSampler(annotations)
        indices = sampler.get_balanced_indices(num_samples)

        # Should still return requested number
        assert len(indices) == num_samples
        # All indices should be valid
        assert all(0 <= idx < len(annotations) for idx in indices)

    def _create_complete_annotations(self, total: int) -> list:
        """Create annotations with all category combinations."""
        categories = [
            ("high", "textual"),
            ("high", "symbolic"),
            ("low", "textual"),
            ("low", "symbolic"),
        ]

        annotations: list[dict] = []
        for _ in range(total):
            vis, msg_type = categories[len(annotations) % 4]
            annotations.append(
                {
                    "visibility_level": vis,
                    "message_type": msg_type,
                }
            )

        random.shuffle(annotations)
        return annotations
