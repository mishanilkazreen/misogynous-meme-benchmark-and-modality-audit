import os
import unittest
from unittest.mock import MagicMock, patch

from torch import nn

from autobenchmark.image_classification import (
    get_device,
    get_feature_extractor,
    get_fine_tuned_model,
    get_transforms,
    scan_dataset_directory,
)


class TestImageClassification(unittest.TestCase):
    def test_get_device(self):
        # Test default auto behavior
        dev = get_device("auto")
        self.assertIn(dev.type, ["cuda", "cpu"])

        # Test explicit cpu
        dev_cpu = get_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

    @patch("torchvision.models.resnet18")
    def test_get_feature_extractor_resnet18(self, mock_resnet18):
        mock_model = MagicMock()
        mock_resnet18.return_value = mock_model

        model, emb_dim = get_feature_extractor("resnet18", pretrained=True)

        mock_resnet18.assert_called_once_with(weights="DEFAULT")
        self.assertIsInstance(model.fc, nn.Identity)
        self.assertEqual(emb_dim, 512)

    @patch("torchvision.models.mobilenet_v3_small")
    def test_get_fine_tuned_model_mobilenet(self, mock_mobilenet):
        # MobileNet has a classifier sequence. We mock it as an nn.Sequential with 4 elements
        mock_model = MagicMock()
        mock_classifier = MagicMock(spec=nn.Sequential)
        mock_layer3 = MagicMock(spec=nn.Linear)
        mock_layer3.in_features = 1024

        # Mock children and classifier structure
        mock_classifier.children.return_value = [None, None, None, mock_layer3]
        mock_model.classifier = mock_classifier
        mock_mobilenet.return_value = mock_model

        model = get_fine_tuned_model("mobilenet_v3_small", num_classes=5, pretrained=True)

        mock_mobilenet.assert_called_once_with(weights="DEFAULT")
        # Check that the 4th element (index 3) is updated to a Linear layer with 5 classes
        self.assertIsInstance(model.classifier[3], nn.Linear)
        self.assertEqual(model.classifier[3].out_features, 5)

    def test_get_transforms(self):
        tf = get_transforms(image_size=128, is_training=False)
        self.assertIsNotNone(tf)

        tf_train = get_transforms(
            image_size=128,
            is_training=True,
            aug_cfg={"random_horizontal_flip": True, "random_rotation_degrees": 10},
        )
        self.assertIsNotNone(tf_train)

    def test_scan_dataset_directory(self):
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            # Create subdirectories for classes
            class_a = os.path.join(temp_dir, "class_a")
            class_b = os.path.join(temp_dir, "class_b")
            os.makedirs(class_a)
            os.makedirs(class_b)

            # Create dummy images
            with open(os.path.join(class_a, "img1.png"), "w") as f:
                f.write("dummy")
            with open(os.path.join(class_b, "img2.jpg"), "w") as f:
                f.write("dummy")

            data, classes = scan_dataset_directory(temp_dir)
            self.assertEqual(classes, ["class_a", "class_b"])
            self.assertEqual(len(data), 2)
            self.assertIn("class_a", [x["label"] for x in data])
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
