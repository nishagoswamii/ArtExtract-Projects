"""
Shared utility functions for GSoC 2025 ArtExtract evaluation tasks.
Includes visualization helpers, metric computation, and data loading utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────

def set_plot_style():
    """Set a consistent, publication-quality plot style."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',
    })
    sns.set_palette("husl")


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix",
                          figsize=(14, 12), normalize=True, top_n=None):
    """
    Plot a confusion matrix heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        figsize: Figure size tuple
        normalize: Whether to normalize the matrix
        top_n: If set, only show the top N most frequent classes
    """
    if top_n and len(class_names) > top_n:
        # Filter to top N most frequent classes
        counter = Counter(y_true)
        top_classes = [c for c, _ in counter.most_common(top_n)]
        mask = np.isin(y_true, top_classes)
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]
        class_names = [class_names[i] for i in sorted(top_classes)]
    
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=len(class_names) <= 20, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_training_history(history, title="Training History"):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dict with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_class_distribution(labels, class_names, title="Class Distribution",
                           figsize=(14, 6), top_n=30):
    """Plot the distribution of classes as a bar chart."""
    counter = Counter(labels)
    if top_n and len(counter) > top_n:
        most_common = counter.most_common(top_n)
    else:
        most_common = counter.most_common()
    
    names = [class_names[idx] if idx < len(class_names) else str(idx) 
             for idx, _ in most_common]
    counts = [c for _, c in most_common]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(names)), counts, color=sns.color_palette("husl", len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Count')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_tsne_embeddings(embeddings, labels, class_names=None, title="t-SNE Visualization",
                         figsize=(14, 10), max_classes=15, perplexity=30, n_samples=5000):
    """
    Create t-SNE visualization of embeddings colored by labels.
    
    Args:
        embeddings: numpy array of shape (N, D)
        labels: numpy array of shape (N,)
        class_names: Optional list mapping label indices to names
        title: Plot title
        max_classes: Maximum number of classes to display
        perplexity: t-SNE perplexity parameter
        n_samples: Maximum number of samples to plot (for speed)
    """
    # Subsample if needed
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # Filter to top classes
    counter = Counter(labels)
    top_classes = [c for c, _ in counter.most_common(max_classes)]
    mask = np.isin(labels, top_classes)
    embeddings = embeddings[mask]
    labels = labels[mask]
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names and label < len(class_names) else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=name,
                  s=10, alpha=0.6)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    return fig


def plot_image_grid(images, titles=None, figsize=(16, 8), nrows=2, ncols=5):
    """Display a grid of images with optional titles."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
                # Denormalize if needed
                if img.min() < 0:
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
            ax.imshow(img)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9, wrap=True)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_similarity_results(query_img, similar_imgs, similarities, query_title="Query",
                            similar_titles=None, figsize=(18, 4)):
    """Display a query image alongside its most similar images."""
    n = len(similar_imgs) + 1
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    # Query image
    if isinstance(query_img, torch.Tensor):
        query_img = query_img.permute(1, 2, 0).cpu().numpy()
        if query_img.min() < 0:
            query_img = query_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        query_img = np.clip(query_img, 0, 1)
    
    axes[0].imshow(query_img)
    axes[0].set_title(f"QUERY\n{query_title}", fontsize=10, fontweight='bold', color='blue')
    axes[0].axis('off')
    
    # Similar images
    for i, (img, sim) in enumerate(zip(similar_imgs, similarities)):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            if img.min() < 0:
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
        
        axes[i + 1].imshow(img)
        title = f"Sim: {sim:.3f}"
        if similar_titles and i < len(similar_titles):
            title = f"{similar_titles[i]}\n{title}"
        axes[i + 1].set_title(title, fontsize=9)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────

def compute_classification_metrics(y_true, y_pred, class_names=None, top_k_acc=5):
    """
    Compute comprehensive classification metrics.
    
    Returns:
        Dict with accuracy, top-k accuracy, f1, precision, recall, per-class report
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names,
                                       zero_division=0, output_dict=True)
        metrics['per_class_report'] = report
    
    return metrics


def compute_topk_accuracy(logits, targets, k=5):
    """Compute top-K accuracy from model logits."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    _, topk_preds = logits.topk(k, dim=1)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds))
    return correct.any(dim=1).float().mean().item()


def find_outliers(logits, true_labels, class_names, threshold=0.8, top_n=20):
    """
    Find potential outliers — samples where the model is very confident 
    about a DIFFERENT class than the true label.
    
    Args:
        logits: Model output logits (N, C)
        true_labels: True labels (N,)
        class_names: List of class names
        threshold: Confidence threshold for flagging
        top_n: Number of top outliers to return
    
    Returns:
        List of dicts with outlier information
    """
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    pred_labels = probs.argmax(axis=1)
    pred_confidence = probs.max(axis=1)
    
    outliers = []
    for i in range(len(true_labels)):
        if pred_labels[i] != true_labels[i] and pred_confidence[i] > threshold:
            outliers.append({
                'index': i,
                'true_label': int(true_labels[i]),
                'true_class': class_names[true_labels[i]] if true_labels[i] < len(class_names) else str(true_labels[i]),
                'pred_label': int(pred_labels[i]),
                'pred_class': class_names[pred_labels[i]] if pred_labels[i] < len(class_names) else str(pred_labels[i]),
                'confidence': float(pred_confidence[i]),
            })
    
    # Sort by confidence (highest first)
    outliers.sort(key=lambda x: x['confidence'], reverse=True)
    return outliers[:top_n]


def compute_retrieval_metrics(query_labels, retrieved_labels, k_values=[1, 5, 10, 20]):
    """
    Compute retrieval metrics for similarity search.
    
    Args:
        query_labels: Labels of query images
        retrieved_labels: 2D array of labels for retrieved images (N_queries, K)
        k_values: Values of K to compute Precision@K
    
    Returns:
        Dict with precision@k, mAP, nDCG
    """
    metrics = {}
    n_queries = len(query_labels)
    
    for k in k_values:
        if k > retrieved_labels.shape[1]:
            continue
        
        # Precision@K: fraction of top-K retrieved items that share the same label
        precisions = []
        for i in range(n_queries):
            relevant = (retrieved_labels[i, :k] == query_labels[i]).sum()
            precisions.append(relevant / k)
        metrics[f'precision@{k}'] = np.mean(precisions)
    
    # Mean Average Precision
    aps = []
    for i in range(n_queries):
        relevant = (retrieved_labels[i] == query_labels[i])
        if relevant.sum() == 0:
            aps.append(0.0)
            continue
        
        cumsum = np.cumsum(relevant)
        precision_at_k = cumsum / (np.arange(len(relevant)) + 1)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap)
    metrics['mAP'] = np.mean(aps)
    
    # nDCG
    ndcgs = []
    for i in range(n_queries):
        relevant = (retrieved_labels[i] == query_labels[i]).astype(float)
        dcg = np.sum(relevant / np.log2(np.arange(2, len(relevant) + 2)))
        ideal = np.sort(relevant)[::-1]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
        ndcgs.append(dcg / (idcg + 1e-8))
    metrics['nDCG'] = np.mean(ndcgs)
    
    return metrics


# ─────────────────────────────────────────────
# Data Helpers
# ─────────────────────────────────────────────

def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU — training will be slow")
    return device


def count_parameters(model):
    """Count trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable
