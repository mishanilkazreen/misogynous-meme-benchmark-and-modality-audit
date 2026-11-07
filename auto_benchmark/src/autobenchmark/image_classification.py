"""
Image classification benchmarking module for the autobenchmark package.
Supports:
1. End-to-end deep learning fine-tuning of pretrained networks.
2. Embedding extraction and traditional machine learning classification suite.
3. Embedding caching with change detection.
"""

import os
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def get_device(device_setting="auto"):
    """Resolve the torch device to use."""
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def get_feature_extractor(model_name, pretrained=True):
    """
    Load a torchvision model and replace its classification head with an Identity layer
    so it functions as a static feature/embedding extractor.

    Returns:
        model: nn.Module
        emb_dim: int (embedding dimension)
    """
    model_name = model_name.lower()
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        emb_dim = 512
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Identity()
        emb_dim = 512
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        emb_dim = 2048
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = nn.Identity()
        emb_dim = 576
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier = nn.Identity()
        emb_dim = 960
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        emb_dim = 1280
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=weights)
        model.classifier = nn.Identity()
        emb_dim = 1280
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Identity()
        emb_dim = 1024
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        model.heads = nn.Identity()
        emb_dim = 768
    else:
        raise ValueError(f"Unsupported backbone model: {model_name}")

    return model, emb_dim


def get_fine_tuned_model(model_name, num_classes, pretrained=True):
    """
    Load a torchvision model and replace its classification head with a Linear
    layer matching the number of classes for end-to-end fine-tuning.
    """
    model_name = model_name.lower()
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        classifier_list = list(model.classifier.children())
        classifier_list[3] = nn.Linear(in_features, num_classes)
        model.classifier = nn.Sequential(*classifier_list)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        classifier_list = list(model.classifier.children())
        classifier_list[3] = nn.Linear(in_features, num_classes)
        model.classifier = nn.Sequential(*classifier_list)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        classifier_list = list(model.classifier.children())
        classifier_list[1] = nn.Linear(in_features, num_classes)
        model.classifier = nn.Sequential(*classifier_list)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        classifier_list = list(model.classifier.children())
        classifier_list[1] = nn.Linear(in_features, num_classes)
        model.classifier = nn.Sequential(*classifier_list)
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model for fine-tuning: {model_name}")

    return model


def get_transforms(image_size=224, is_training=False, aug_cfg=None):
    """Get transformation pipeline for vision models."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_training and aug_cfg:
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        if aug_cfg.get("random_horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip())

        rot_deg = aug_cfg.get("random_rotation_degrees", 0)
        if rot_deg > 0:
            transform_list.append(transforms.RandomRotation(rot_deg))

        if aug_cfg.get("color_jitter", False):
            transform_list.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )

        transform_list.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(transform_list)
    else:
        return transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
        )


def scan_dataset_directory(dataset_dir):
    """
    Scan a directory structure:
    dataset_dir/
      class_A/
        img1.png
        ...
      class_B/
        img2.jpg
        ...
    Returns:
        data: List of dicts, e.g. [{'path': absolute_path, 'label': class_name, 'mtime': mod_time}]
        classes: List of sorted class names
    """
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    data = []

    # We resolve train/val structure or flat directory.
    # Check if there are splits: train/val inside dataset_dir
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(dataset_dir, "validation")

    is_pre_split = os.path.exists(train_dir)

    classes_set = set()

    def process_dir(target_dir):
        subdirs = sorted(
            [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        )
        for subdir in subdirs:
            classes_set.add(subdir)
            class_path = os.path.join(target_dir, subdir)
            for root, _, files in os.walk(class_path):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in allowed_extensions:
                        full_path = os.path.join(root, f)
                        try:
                            mtime = os.path.getmtime(full_path)
                            data.append(
                                {
                                    "path": os.path.abspath(full_path),
                                    "label": subdir,
                                    "mtime": mtime,
                                }
                            )
                        except Exception:
                            pass

    if is_pre_split:
        process_dir(train_dir)
        if os.path.exists(val_dir):
            process_dir(val_dir)
    else:
        process_dir(dataset_dir)

    return data, sorted(classes_set)


class ImageClassificationDataset(Dataset):
    """Custom Dataset loading images from file paths."""

    def __init__(self, file_list, class_to_idx, transform=None):
        self.file_list = file_list
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]
        img_path = item["path"]
        label_name = item["label"]
        label_idx = self.class_to_idx[label_name]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback: create a dummy tensor of shape 3x224x224
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            img = self.transform(img)

        return img, label_idx


def get_embeddings_with_cache(dataset_dir, backbone_model_name, image_size, device, use_cache=True):
    """
    Extract features/embeddings for all images in the directory structure.
    Saves embeddings to a cached file to allow instant loads in subsequent runs.
    """
    scanned_files, classes = scan_dataset_directory(dataset_dir)
    if not scanned_files:
        raise ValueError(f"No valid images found in directory: {dataset_dir}")

    # Create cache folder under dataset_dir parent folder
    parent_dir = os.path.dirname(os.path.abspath(dataset_dir))
    cache_dir = os.path.join(parent_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Formulate cache filename based on directory name, backbone, and image size
    safe_backbone_name = backbone_model_name.replace("/", "_").replace(":", "_")
    dir_name = os.path.basename(os.path.abspath(dataset_dir))
    cache_filename = f"{dir_name}_{safe_backbone_name}_{image_size}_embeddings.joblib"
    cache_path = os.path.join(cache_dir, cache_filename)

    load_from_cache = False
    cached_data = None

    if use_cache and os.path.exists(cache_path):
        try:
            print(f"Loading cached embeddings from: {cache_path}")
            cached_data = joblib.load(cache_path)
            cached_files_dict = cached_data.get("files_metadata", {})

            # Match sizes and modification times
            if len(scanned_files) == len(cached_files_dict):
                match = True
                for item in scanned_files:
                    path = item["path"]
                    if path not in cached_files_dict:
                        match = False
                        break
                    # Check modified time and label
                    if (
                        cached_files_dict[path]["mtime"] != item["mtime"]
                        or cached_files_dict[path]["label"] != item["label"]
                    ):
                        match = False
                        break
                if match:
                    load_from_cache = True
                    print("Cache hit! Embeddings loaded successfully.")
                else:
                    print("Cache stale (files modified). Re-extracting embeddings...")
            else:
                print("Cache stale (number of files changed). Re-extracting embeddings...")
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}. Re-extracting...")

    if load_from_cache and cached_data is not None:
        return cached_data["embeddings"], cached_data["labels"], classes

    # Extract features
    print(f"Extracting static image features using {backbone_model_name} on {device}...")
    from tqdm import tqdm

    model, _emb_dim = get_feature_extractor(backbone_model_name, pretrained=True)
    model = model.to(device)
    model.eval()

    transform = get_transforms(image_size, is_training=False)

    embeddings = {}
    labels = {}

    batch_size = 32
    batches = [scanned_files[i : i + batch_size] for i in range(0, len(scanned_files), batch_size)]

    with torch.no_grad():
        for batch in tqdm(batches, desc="Image Feature Extraction"):
            imgs = []
            valid_indices = []

            for idx, item in enumerate(batch):
                try:
                    img = Image.open(item["path"]).convert("RGB")
                    tensor = transform(img)
                    imgs.append(tensor)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Warning: Failed to load image {item['path']}: {e}")

            if not imgs:
                continue

            batch_tensors = torch.stack(imgs).to(device)
            outputs = model(batch_tensors)

            # Average pooling if output is 4D (conv models output [N, C, H, W] if classifier is Identity)
            if len(outputs.shape) == 4:
                outputs = outputs.mean(dim=[2, 3])

            outputs_np = outputs.cpu().numpy()

            for idx, val_idx in enumerate(valid_indices):
                item = batch[val_idx]
                embeddings[item["path"]] = outputs_np[idx]
                labels[item["path"]] = item["label"]

    # Save cache
    if use_cache:
        files_metadata = {
            item["path"]: {"mtime": item["mtime"], "label": item["label"]} for item in scanned_files
        }
        save_data = {
            "files_metadata": files_metadata,
            "embeddings": embeddings,
            "labels": labels,
            "classes": classes,
            "backbone": backbone_model_name,
            "image_size": image_size,
        }
        try:
            joblib.dump(save_data, cache_path)
            print(f"Saved embeddings cache -> {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    return embeddings, labels, classes


def train_deep_model(
    model_spec, data_cfg, train_loader, val_loader, num_classes, output_dir, device
):
    """
    Trains a single deep model using cross entropy loss and saves the best model.
    """
    model_name = model_spec["name"]
    model_key = model_spec["model_name"]
    pretrained = model_spec.get("pretrained", True)

    print(f"\nTraining Deep Model: {model_name}")
    try:
        model = get_fine_tuned_model(model_key, num_classes, pretrained=pretrained)
    except Exception as e:
        print(f"  ERROR: Failed to load model {model_key}: {e}")
        return {
            "Model": model_name,
            "Status": f"FAILED: {e}",
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "Training_Time": 0.0,
        }

    model = model.to(device)

    # Train parameters
    train_cfg = data_cfg.get("training", {})
    epochs = train_cfg.get("epochs", 10)
    lr = train_cfg.get("learning_rate", 0.001)
    opt_name = train_cfg.get("optimizer", "AdamW")

    criterion = nn.CrossEntropyLoss()

    if opt_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    best_model_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_best.pt")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total if total > 0 else 0.0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"  Epoch {epoch}/{epochs} - Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)

    training_time = time.time() - start_time
    print(f"  Training finished in {training_time:.1f}s. Best Accuracy: {best_acc:.4f}")

    # Load best model for evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    result = {
        "Model": model_name,
        "Status": "OK",
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Training_Time": training_time,
    }

    # Save training history chart
    _plot_training_history(history, model_name, output_dir)

    return result


def _plot_training_history(history, model_name, output_dir):
    """Plot epochs vs training/validation curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} Loss History")
    ax1.legend()

    # Accuracy curves
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="forestgreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} Accuracy History")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()


def run_classification_benchmark(data_cfg, model_cfg, output_dir, init_config=None):
    """
    Runs the image classification benchmark according to the mode:
    - "extract_features": Extracts features and feeds them into the traditional ML pipeline in models.py.
    - "fine_tune": Runs deep learning fine-tuning.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_cfg = data_cfg.get("dataset", {})
    dataset_dir = dataset_cfg.get("dataset_dir", "")

    # Handle base_dir resolution if not absolute path
    if not os.path.isabs(dataset_dir) and init_config:
        base_dir = init_config.get("system", {}).get("base_dir", "")
        if base_dir:
            full_path = os.path.join(base_dir, dataset_dir)
            if os.path.exists(full_path):
                dataset_dir = full_path

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    mode = data_cfg.get("mode", "extract_features")
    image_size = dataset_cfg.get("image_size", 224)
    model_config_name = model_cfg.get("config_name", "image_classification")

    # Resolve device
    train_cfg = data_cfg.get("training", {})
    device = get_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device} for image classification benchmark.")

    if mode == "extract_features":
        # 1. Feature extraction + Traditional ML
        backbone_model = data_cfg.get("backbone_model", "resnet50")
        use_cache = data_cfg.get("use_cache", True)

        # Extract embeddings
        embeddings, labels, classes = get_embeddings_with_cache(
            dataset_dir, backbone_model, image_size, device, use_cache=use_cache
        )

        # 2. Re-arrange into train-val splits based on directory structure or validation_split
        # Check if directories 'train' and 'val' exist
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")
        if not os.path.exists(val_dir):
            val_dir = os.path.join(dataset_dir, "validation")

        is_pre_split = os.path.exists(train_dir)

        X_paths = sorted(embeddings.keys())
        X = np.array([embeddings[p] for p in X_paths])
        y = np.array([labels[p] for p in X_paths])

        if is_pre_split:
            train_paths = []
            val_paths = []
            for p in X_paths:
                rel = os.path.relpath(p, dataset_dir)
                parts = rel.split(os.sep)
                if len(parts) > 1 and parts[0] == "train":
                    train_paths.append(p)
                elif len(parts) > 1 and parts[0] in ("val", "validation"):
                    val_paths.append(p)

            X_train = np.array([embeddings[p] for p in train_paths])
            y_train = np.array([labels[p] for p in train_paths])
            X_test = np.array([embeddings[p] for p in val_paths])
            y_test = np.array([labels[p] for p in val_paths])
            print("Dataset splits resolved from folder structure:")
            print(f"  Train: {len(X_train)} samples, Val: {len(X_test)} samples")
        else:
            from sklearn.model_selection import train_test_split

            val_split = dataset_cfg.get("validation_split", 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=val_split, random_state=42, stratify=y
            )
            print(f"Dataset split randomly (val_split={val_split}):")
            print(f"  Train: {len(X_train)} samples, Val: {len(X_test)} samples")

        # 3. Apply standard scaling to features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler for future runs
        scaler_path = os.path.join(output_dir, f"{model_config_name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)

        # 4. Invoke traditional model benchmarking
        from autobenchmark.evaluation import save_and_rank_results
        from autobenchmark.models import train_benchmark_models

        # Set up parameters in model_cfg
        model_cfg["models_to_run"] = model_cfg.get("traditional_models", "all")

        # Create a list of feature names: feat_0, feat_1, ...
        feat_labels = [f"feat_{i}" for i in range(X_train.shape[1])]

        # Fit models
        df_results, _predictions_df = train_benchmark_models(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            model_cfg,
            output_dir,
            feat_labels=feat_labels,
        )

        # Rank models
        save_and_rank_results(df_results, model_cfg, output_dir)
        return df_results

    elif mode == "fine_tune":
        # 1. Deep model training end-to-end
        scanned_files, classes = scan_dataset_directory(dataset_dir)
        num_classes = len(classes)
        print(f"Fine-tuning classification benchmark. Classes found ({num_classes}): {classes}")

        class_to_idx = {name: i for i, name in enumerate(classes)}

        # Split files into train / val
        train_dir = os.path.join(dataset_dir, "train")
        is_pre_split = os.path.exists(train_dir)

        if is_pre_split:
            train_files = []
            val_files = []
            for item in scanned_files:
                rel = os.path.relpath(item["path"], dataset_dir)
                parts = rel.split(os.sep)
                if len(parts) > 1 and parts[0] == "train":
                    train_files.append(item)
                elif len(parts) > 1 and parts[0] in ("val", "validation"):
                    val_files.append(item)
        else:
            # Random split
            np.random.seed(42)
            shuffled = list(scanned_files)
            np.random.shuffle(shuffled)
            val_split = dataset_cfg.get("validation_split", 0.2)
            split_idx = int(len(shuffled) * (1 - val_split))
            train_files = shuffled[:split_idx]
            val_files = shuffled[split_idx:]

        print(f"Splits: Train={len(train_files)} images, Val={len(val_files)} images")

        # Create Dataloaders
        aug_cfg = data_cfg.get("augmentation", {})
        train_transform = get_transforms(image_size, is_training=True, aug_cfg=aug_cfg)
        val_transform = get_transforms(image_size, is_training=False)

        train_dataset = ImageClassificationDataset(
            train_files, class_to_idx, transform=train_transform
        )
        val_dataset = ImageClassificationDataset(val_files, class_to_idx, transform=val_transform)

        batch_size = train_cfg.get("batch_size", 16)
        workers = train_cfg.get("workers", 0)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
        )

        deep_models = model_cfg.get("deep_models", [])
        if not deep_models:
            print("Warning: No deep models specified under 'deep_models' in model config.")
            return pd.DataFrame()

        results_list = []
        for model_spec in deep_models:
            res = train_deep_model(
                model_spec, data_cfg, train_loader, val_loader, num_classes, output_dir, device
            )
            results_list.append(res)

        df_results = pd.DataFrame(results_list)

        # Save evaluation results CSV
        from autobenchmark.evaluation import save_and_rank_results

        save_and_rank_results(df_results, model_cfg, output_dir)
        return df_results

    else:
        raise ValueError(f"Unknown image classification mode: {mode}")
