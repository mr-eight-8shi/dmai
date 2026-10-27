import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from tqdm import tqdm


class PixelVectorExtractor(nn.Module):
    """[Raw Pixel-based] Extractor that normalizes images directly into feature vectors."""
    def __init__(self, target_dim=512):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        batch_size = x.size(0)
        flat = x.view(batch_size, -1)
        return nn.functional.normalize(flat, p=2, dim=1)


class SmallLocalExpertCNN(nn.Module):
    """Lightweight general-purpose CNN classifier corresponding to each vector cluster."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)


class DynamicClusterRouterPipeline:
    """
    Sequential clustering pipeline equipped with raw-pixel vector extraction,
    model caching, and real-time similarity search.
    """
    def __init__(self, config=None):
        self.config = config
        
        self.space_dir = config.space_dir
        self.device = config.device
        
        self.similarity_threshold = config.router.similarity_threshold
        self.training_trigger_count = config.router.training_trigger_count
        self.max_samples_per_class = getattr(config.router, "max_samples_per_class", 300)

        self.clusters_dir = os.path.join(self.space_dir, "expert_models")
        self.data_dir = os.path.join(self.space_dir, "expert_datasets")
        self.state_file = os.path.join(self.space_dir, "clusters_state.json")
        
        os.makedirs(self.clusters_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.encoder = PixelVectorExtractor().to(self.device)
        self.encoder.eval()

        self.cluster_cache = {} 
        self.cluster_buffers = {} 
        self.model_cache = {}     
        self.cluster_ids_list = []
        self.cluster_matrix = None

        self._load_state_from_json()

    def _load_state_from_json(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for c_id, info in data.items():
                        raw_class_counts = info.get("class_counts", {})
                        class_counts = {int(k): v for k, v in raw_class_counts.items()}

                        self.cluster_cache[c_id] = {
                            "sample_count": info["sample_count"],
                            "samples_since_last_train": info.get("samples_since_last_train", 0),
                            "class_counts": class_counts,
                            "image_shape": info.get("image_shape", []),
                            "classes": set(info["classes"]),
                            "representative_vector": np.array(info["representative_vector"], dtype=np.float32),
                            "accuracy": info.get("accuracy", 0.0)
                        }
                        self.cluster_buffers[c_id] = {"images": [], "labels": []}
                print(f"📂 Loaded existing cluster state from JSON (Total clusters: {len(self.cluster_cache)})")
                self._update_cluster_matrix()
            except Exception as e:
                print(f"[Warning] Failed to load JSON state: {e}")

    def _update_cluster_matrix(self):
        self.cluster_ids_list = sorted(list(self.cluster_cache.keys()))
        if not self.cluster_ids_list:
            self.cluster_matrix = None
            return
        
        vecs = [self.cluster_cache[c_id]["representative_vector"] for c_id in self.cluster_ids_list]
        self.cluster_matrix = np.vstack(vecs)

    def _save_state_to_json(self):
        serializable_data = {}
        for c_id, info in self.cluster_cache.items():
            serializable_data[c_id] = {
                "sample_count": info["sample_count"],
                "samples_since_last_train": info["samples_since_last_train"],
                "class_counts": {str(k): v for k, v in info["class_counts"].items()},
                "image_shape": list(info.get("image_shape", [])),
                "classes": sorted(list(info["classes"])),
                "representative_vector": info["representative_vector"].tolist(),
                "accuracy": info.get("accuracy", 0.0)
            }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        self._update_cluster_matrix()

    def _get_model(self, cluster_id: str):
        if cluster_id in self.model_cache:
            return self.model_cache[cluster_id]

        model_path = os.path.join(self.clusters_dir, f"{cluster_id}.pth")
        if not os.path.exists(model_path):
            return None

        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        model = SmallLocalExpertCNN(num_classes=ckpt["num_classes"]).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self.model_cache[cluster_id] = model
        return model

    def _search_similar_clusters_numpy(self, query_vec: np.ndarray, limit: int = 3) -> list:
        if self.cluster_matrix is None or len(self.cluster_ids_list) == 0:
            return []

        sims = self.cluster_matrix @ query_vec

        candidates = []
        for idx, sim in enumerate(sims):
            c_id = self.cluster_ids_list[idx]
            info = self.cluster_cache[c_id]
            candidates.append({
                "model_id": c_id,
                "sample_count": info["sample_count"],
                "samples_since_last_train": info["samples_since_last_train"],
                "image_shape": info.get("image_shape", []),
                "score": float(sim)
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:limit]

    def _extract_vector(self, img_np: np.ndarray) -> np.ndarray:
        if len(img_np.shape) == 2:
            img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
        
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            vec = self.encoder(img_tensor).cpu().numpy()[0]
        return vec

    def register_or_route_image(self, img_np, label: int):
        vec = self._extract_vector(img_np)
        img_shape = list(img_np.shape)
        
        similar_clusters = self._search_similar_clusters_numpy(vec, limit=1)
        best_match = similar_clusters[0] if similar_clusters else None
        highest_sim = best_match["score"] if best_match else -1.0
        best_match_cluster_id = best_match["model_id"] if best_match else None

        if best_match_cluster_id and highest_sim >= self.similarity_threshold:
            appended = self._append_to_cluster(best_match_cluster_id, img_np, label, vec)
            if appended:
                return best_match_cluster_id, "appended"
            else:
                return best_match_cluster_id, "ignored_max_per_class"

        # Create a new general model cluster if no match is found
        new_cluster_id = f"cluster_{len(self.cluster_cache) + 1:03d}"
        self.cluster_cache[new_cluster_id] = {
            "sample_count": 0,
            "samples_since_last_train": 0,
            "class_counts": {},
            "image_shape": img_shape,
            "classes": set(),
            "representative_vector": vec,
            "accuracy": 0.0
        }
        self.cluster_buffers[new_cluster_id] = {"images": [], "labels": []}
        
        # Immediately update matrix so subsequent searches can reference the new cluster
        self._update_cluster_matrix()
        
        self._append_to_cluster(new_cluster_id, img_np, label, vec)
        return new_cluster_id, "created"

    def _append_to_cluster(self, cluster_id: str, img_np: np.ndarray, label: int, new_vec: np.ndarray):
        c_info = self.cluster_cache[cluster_id]
        
        if cluster_id not in self.cluster_buffers:
            self.cluster_buffers[cluster_id] = {"images": [], "labels": []}
        
        class_counts = c_info["class_counts"]
        
        current_class_count = class_counts.get(label, 0)
        if current_class_count >= self.max_samples_per_class:
            return False

        self.cluster_buffers[cluster_id]["images"].append(img_np)
        self.cluster_buffers[cluster_id]["labels"].append(label)

        class_counts[label] = current_class_count + 1
        c_info["classes"].add(label)
        c_info["sample_count"] += 1
        c_info["samples_since_last_train"] += 1

        # Update representative vector incrementally
        current_total = c_info["sample_count"]
        old_mean_vec = c_info["representative_vector"]
        alpha = 0.05 if current_total > 1 else 1.0
        mean_vec = (1.0 - alpha) * old_mean_vec + alpha * new_vec
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        c_info["representative_vector"] = mean_vec

        self._update_cluster_matrix()

        if current_total % 100 == 0:
            self._save_state_to_json()

        if c_info["samples_since_last_train"] >= self.training_trigger_count:
            self.flush_and_train_local_classifier(cluster_id)
            
        return True

    def flush_and_train_local_classifier(self, cluster_id: str, max_epochs: int = 30, patience: int = 5):
        c_info = self.cluster_cache[cluster_id]
        buf = self.cluster_buffers.get(cluster_id, {"images": [], "labels": []})
        
        images = buf["images"]
        labels = buf["labels"]
        if not images:
            return

        cluster_data_dir = os.path.join(self.data_dir, cluster_id)
        os.makedirs(cluster_data_dir, exist_ok=True)
        
        X = np.array(images, dtype=np.float32)
        if len(X.shape) == 3:
            X = X[:, np.newaxis, :, :]
        y = np.array(labels, dtype=np.int64)

        np.save(os.path.join(cluster_data_dir, "images.npy"), X)
        np.save(os.path.join(cluster_data_dir, "labels.npy"), y)

        self.cluster_buffers[cluster_id] = {"images": [], "labels": []}
        c_info["samples_since_last_train"] = 0

        unique_classes = np.unique(y)
        num_classes = max(10, int(np.max(unique_classes) + 1))

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = SmallLocalExpertCNN(num_classes=num_classes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        augment_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        ])

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"\n🚀 [Cluster {cluster_id}] Starting model training due to reaching sample trigger limit (Max Epochs: {max_epochs})")
        
        epoch_pbar = tqdm(range(max_epochs), desc=f"Training {cluster_id}")
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                b_x = augment_transform(b_x)

                optimizer.zero_grad()
                out = model(b_x)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
            train_loss /= train_size

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for b_x, b_y in val_loader:
                    b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                    out = model(b_x)
                    loss = criterion(out, b_y)
                    val_loss += loss.item() * b_x.size(0)
            val_loss /= val_size if val_size > 0 else 1

            epoch_pbar.set_postfix({"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        full_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for b_x, b_y in full_dataloader:
                b_x, b_y = b_x.to(self.device), b_y.to(self.device)
                preds = torch.argmax(model(b_x), dim=1)
                correct += (preds == b_y).sum().item()
                total += b_y.size(0)
        acc = correct / total if total > 0 else 0.0

        model_save_path = os.path.join(self.clusters_dir, f"{cluster_id}.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "classes": unique_classes.tolist()
        }, model_save_path)

        c_info["accuracy"] = acc
        
        if cluster_id in self.model_cache:
            del self.model_cache[cluster_id]

        self._save_state_to_json()

    def simulate_streaming_ingestion(self, X_train, y_train, num_samples: int = -1):
        print(f"\n=== [Streaming Ingestion & Clustering Simulation] ===")
        
        total_data_len = len(X_train)
        indices = np.random.permutation(total_data_len) if num_samples == -1 else np.arange(min(num_samples, total_data_len))
            
        actual_run_count = len(indices)
        print(f"📥 Total samples to process: {actual_run_count}\n")

        for i, idx in enumerate(indices):
            img = X_train[idx]
            lbl = int(y_train[idx])
            
            self.register_or_route_image(img, lbl)
            
            if (i + 1) % 200 == 0 or (i + 1) == actual_run_count:
                self._print_live_dashboard(i + 1, actual_run_count)

        self._save_state_to_json()

    def _print_live_dashboard(self, current_step: int, total_steps: int):
        print(f"\n📊 [Streaming Progress: {current_step}/{total_steps}]")
        print("================================================================================")
        for c_id in sorted(self.cluster_cache.keys()):
            c_info = self.cluster_cache[c_id]
            count = c_info["sample_count"]
            since_train = c_info["samples_since_last_train"]
            shape = c_info.get("image_shape", [])
            classes = sorted(list(c_info["classes"]))
            acc = c_info.get("accuracy", 0.0)
            
            status_str = f"ACTIVE (acc {acc*100:.1f}%, unfitted:{since_train}/{self.training_trigger_count})"
            
            print(f" {c_id:<10} Shape:{str(shape):<12} Total:{count:>5}  {status_str:<36} Classes: {classes}")
        print("================================================================================")

    def evaluate_full_testset(self, X_test, y_test, batch_size: int = 256, top_k: int = 3):
        total_count = len(X_test)
        print(f"\n🧪 Starting Top-k Ensemble Evaluation (Total {total_count} samples, Top-k: {top_k})...")
        
        correct_count = 0
        unknown_count = 0
        class_stats = {lbl: {"total": 0, "correct": 0, "unknown": 0} for lbl in range(10)}

        for lbl in y_test:
            class_stats[int(lbl)]["total"] += 1

        for start_idx in range(0, total_count, batch_size):
            end_idx = min(start_idx + batch_size, total_count)
            batch_imgs = X_test[start_idx:end_idx]
            batch_labels = y_test[start_idx:end_idx]

            if len(batch_imgs.shape) == 3:
                batch_tensor = torch.tensor(batch_imgs, dtype=torch.float32).unsqueeze(1).to(self.device)
            else:
                batch_tensor = torch.tensor(batch_imgs, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                batch_vectors = self.encoder(batch_tensor).cpu().numpy()

            for i in range(len(batch_vectors)):
                vec = batch_vectors[i]
                true_lbl = int(batch_labels[i])
                single_tensor = batch_tensor[i:i+1]
                
                similar_clusters = self._search_similar_clusters_numpy(vec, limit=top_k)
                
                weighted_logits = None
                total_sim = 0.0
                valid_match_found = False

                for match in similar_clusters:
                    cluster_id = match["model_id"]
                    similarity = float(match["score"])

                    if similarity < self.similarity_threshold:
                        continue

                    model = self._get_model(cluster_id)
                    if model is None:
                        continue

                    valid_match_found = True
                    with torch.no_grad():
                        output = model(single_tensor)
                        probs = torch.softmax(output, dim=1)

                    if weighted_logits is None:
                        weighted_logits = probs * similarity
                    else:
                        if weighted_logits.size(1) == probs.size(1):
                            weighted_logits += probs * similarity
                    total_sim += similarity

                if not valid_match_found or total_sim == 0.0 or weighted_logits is None:
                    unknown_count += 1
                    class_stats[true_lbl]["unknown"] += 1
                    continue

                final_probs = weighted_logits / total_sim
                pred_class = int(torch.argmax(final_probs, dim=1).item())

                if pred_class == true_lbl:
                    correct_count += 1
                    class_stats[true_lbl]["correct"] += 1

        eval_count = total_count - unknown_count
        overall_acc = (correct_count / eval_count * 100) if eval_count > 0 else 0.0
        total_acc = (correct_count / total_count * 100)

        print("\n" + "=" * 75)
        print("📊 [Test Set Evaluation Results (Top-k Ensemble)]")
        print("=" * 75)
        print(f" • Total Test Samples    : {total_count:>6}")
        print(f" • Correct Predictions   : {correct_count:>6}")
        print(f" • Unknown / Rejected    : {unknown_count:>6} ({unknown_count/total_count*100:.2f}%)")
        print(f" • Real Overall Accuracy : {total_acc:.2f}%")
        print(f" • Evaluated Accuracy    : {overall_acc:.2f}%")
        print("-" * 75)
        print(" 🏷️ [Per-Class Breakdown]")
        print(" Label |  Total  | Correct | Unknown | Accuracy (Evaluated) | Real Accuracy")
        print("-" * 75)
        
        for lbl in range(10):
            st = class_stats[lbl]
            tot = st["total"]
            cor = st["correct"]
            unk = st["unknown"]
            ev_cnt = tot - unk
            
            acc_eval = (cor / ev_cnt * 100) if ev_cnt > 0 else 0.0
            acc_tot = (cor / tot * 100) if tot > 0 else 0.0
            
            print(f"    {lbl}   | {tot:>7} | {cor:>7} | {unk:>7} | {acc_eval:>19.2f}% | {acc_tot:>19.2f}%")
        print("=" * 75 + "\n")