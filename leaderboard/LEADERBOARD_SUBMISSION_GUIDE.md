# AAE5303 - Leaderboard Submission Guide

## 📁 Evaluation Dataset

**UAVScenes HKisland** - 2D Semantic Segmentation

| Resource | Link |
|----------|------|
| UAVScenes GitHub | https://github.com/sijieaaa/UAVScenes |
| MARS-LVIG Dataset | https://mars.hku.hk/dataset.html |

---

## 📊 Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| **Dice Score** | ↑ Higher is better | F1-Score for segmentation (0-100%) |
| **mIoU** | ↑ Higher is better | Mean Intersection over Union (0-100%) |
| **FWIoU** | ↑ Higher is better | Frequency Weighted IoU (0-100%) |

---

## 📄 JSON Submission Format

Submit your results using the following JSON format:

```json
{
    "group_id": "YOUR_GROUP_ID",
    "group_name": "Your Group Name",
    "metrics": {
        "dice_score": 38.54,
        "miou": 32.93,
        "fwiou": 65.21
    },
    "submission_date": "YYYY-MM-DD"
}
```

### Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `group_id` | string | Your group ID | `"Group_01"` |
| `group_name` | string | Your group name | `"Team Alpha"` |
| `metrics.dice_score` | number | Dice Score (%) | `38.54` |
| `metrics.miou` | number | mIoU value (%) | `32.93` |
| `metrics.fwiou` | number | FWIoU value (%) | `65.21` |
| `submission_date` | string | Date (YYYY-MM-DD) | `"2024-12-18"` |

### File Naming

`{GroupID}_leaderboard.json`

Example: `Group_01_leaderboard.json`

---

## 🔧 How to Generate Submission

### Step 1: Train Your Model

```bash
python train.py \
    --epochs 20 \
    --batch-size 2 \
    --learning-rate 0.0001 \
    --scale 0.25
```

### Step 2: Run Evaluation Script

```bash
python evaluate_submission.py \
    --model checkpoints/checkpoint_epoch20.pth \
    --scale 0.25 \
    --output Group_01_leaderboard.json \
    --team "Your Group Name" \
    --group-id "Group_01"
```

### Step 3: Verify JSON Format

Ensure your JSON file matches the required format:

```python
import json

with open('Group_01_leaderboard.json', 'r') as f:
    submission = json.load(f)

# Required fields
assert 'group_id' in submission
assert 'group_name' in submission
assert 'metrics' in submission
assert 'dice_score' in submission['metrics']
assert 'miou' in submission['metrics']
assert 'fwiou' in submission['metrics']
assert 'submission_date' in submission

print("✓ Submission format is valid!")
```

---

## 📊 Baseline Results

| Metric | Baseline Value |
|--------|----------------|
| **Dice Score** | 38.54% |
| **mIoU** | 32.93% |
| **FWIoU** | 65.21% |

**Training Configuration:**
- Epochs: 5
- Batch Size: 2
- Learning Rate: 1e-4
- Scale: 0.25
- Hardware: CPU only

---

## 💡 Tips for Improvement

### Easy (Expected: +10-20% mIoU)
1. Train more epochs (15-20)
2. Adjust learning rate
3. Increase image scale (0.3-0.5)

### Medium (Expected: +20-30% mIoU)
4. Data augmentation (flip, rotate, color jitter)
5. Learning rate scheduler
6. Weighted loss for class imbalance

### Advanced (Expected: +30-40% mIoU)
7. Focal loss
8. Class-balanced sampling
9. Test-time augmentation
10. Ensemble methods

---

## 🌐 Leaderboard Website & Baseline

> **📢 The leaderboard submission website and baseline results will be announced later.**
