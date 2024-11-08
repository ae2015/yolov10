import os, sys
os.environ['YOLO_VERBOSE'] = 'False'
os.chdir("/content/gdrive/MyDrive/code/yolov10") # os.chdir("/Users/alex/Documents/Code/yolov10")
sys.path.append('.')
os.environ['HF_HOME'] = "/content/gdrive/MyDrive/cache/huggingface/datasets"

from ultralytics import settings

# Update the datasets directory setting
settings.update({"datasets_dir": "/content/gdrive/MyDrive/datasets"}) # settings.update({"datasets_dir": "."})

from ultralytics import YOLO  #  Reads settings
from ultralytics.engine.results import Results

import numpy as np
from datasets import load_dataset, DownloadMode
import aiohttp
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

categories = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

def score_summary(pred_summary, true_summary, pred_confidence_threshold = 0.5):
    """
    Computes the maximum-weight bipartite matching between the entries in
    the predicted summary and the entries in the true (ground-truth) summary.
    Matches only same-class entities, weighted by their intersection area
    divided by union area. Once the best match is found, computes per-class
    statistics and returns them.

    Parameters
    ----------
    pred_summary : list
        The list of entries for objects on the image predicted by YOLO.
        Each entry is a dictionary as follows:
        `{'name': str, 'class': int, 'confidence': float, 'box': dict}`
        The box dictionary has four items:
        `{'x1': float, 'y1': float, 'x2': float, y2': float}`
        The format is the same as what's returned by `Results.summary()`

    true_summary : list
        The list of ground-truth entries on the image, annotated by a human.
        If multiple annotations are provided, call once per each annotation.
        The format is the same as in `pred_summary`.

    pred_confidence_threshold : float
        Predicted entities with confidence lower that this will be rejected.

    Returns
    -------
    areas : list
        A list with entries ordered as in `categories`; for each category
        we provide the following dictionary:
        `{'pred_area': float, 'true_area': float, 'intersect_area': float}`
    """
    IA = np.zeros((len(pred_summary), len(true_summary)))  # Matrix for box intersection areas
    Wt = np.zeros((len(pred_summary), len(true_summary)))  # Matrix for weights = intersection over union
    for i, pred_entry in enumerate(pred_summary):
        if pred_entry['confidence'] >= pred_confidence_threshold:
            for j, true_entry in enumerate(true_summary):
                if pred_entry['class'] == true_entry['class']:
                    pred_box = pred_entry['box']
                    true_box = true_entry['box']
                    pred_area = (pred_box['x2'] - pred_box['x1']) * (pred_box['y2'] - pred_box['y1'])
                    true_area = (true_box['x2'] - true_box['x1']) * (true_box['y2'] - true_box['y1'])
                    intersect_box = {
                        'x1' : max(pred_box['x1'], true_box['x1']),
                        'y1' : max(pred_box['y1'], true_box['y1']),
                        'x2' : min(pred_box['x2'], true_box['x2']),
                        'y2' : min(pred_box['y2'], true_box['y2'])
                    }
                    if (intersect_box['x1'] < intersect_box['x2'] and intersect_box['y1'] < intersect_box['y2']):
                        IA[i, j] = (intersect_box['x2'] - intersect_box['x1']) * (intersect_box['y2'] - intersect_box['y1'])
                        Wt[i, j] = IA[i, j] / (pred_area + true_area - IA[i, j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix = Wt, maximize = True)

    areas = [{'pred_area': 0.0, 'true_area': 0.0, 'intersect_area': 0.0} for _ in categories]
    for i, pred_entry in enumerate(pred_summary):
        if pred_entry['confidence'] >= pred_confidence_threshold:
            pred_box = pred_entry['box']
            pred_area = (pred_box['x2'] - pred_box['x1']) * (pred_box['y2'] - pred_box['y1'])
            areas[pred_entry['class']]['pred_area'] += pred_area
    for j, true_entry in enumerate(true_summary):
        true_box = true_entry['box']
        true_area = (true_box['x2'] - true_box['x1']) * (true_box['y2'] - true_box['y1'])
        areas[true_entry['class']]['true_area'] += true_area
    for i, j in zip(row_ind, col_ind):
        areas[pred_summary[i]['class']]['intersect_area'] += IA[i, j]
    return areas

# Load the DocLayNet dataset

dataset = load_dataset(
    "ds4sd/DocLayNet",
    download_mode = DownloadMode.REUSE_DATASET_IF_EXISTS,
    storage_options = {'client_kwargs': {'timeout': aiohttp.ClientTimeout(total = 3600)}},
    trust_remote_code = True
)
print(dataset)

# Load a pre-trained YOLOv10n model fine-tuned on DocLayNet
# From https://huggingface.co/omoured/YOLOv10-Document-Layout-Analysis

model = YOLO("/content/gdrive/MyDrive/code/yolov10/runs/detect/train2/weights/best.pt")
# model = YOLO("huggingface-omoured-DocLayNet-yolov10m_best.pt")
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")

print("Evaluating precision and recall on a subset of records...")

aggregated_areas = [{'pred_area': 0.0, 'true_area': 0.0, 'intersect_area': 0.0} for _ in categories]
aggregated_micro_scores = [{'precision': 0.0, 'recall': 0.0, 'F1': 0.0} for _ in categories]
aggregated_micro_counts = [{'precision': 0, 'recall': 0, 'F1': 0} for _ in categories]

for d_index, datum in tqdm(enumerate(dataset["test"]), total = len(dataset["test"])):
# for d_index in tqdm(range(200), total = 200):
#     datum = dataset["validation"][d_index]

    # Use `datum` to instantiate ground truth summaries, one summary per
    # one human annotation. Each annotation is a complete set of entities for
    # the image. Note that sometimes there are multiple annotations,
    # in which case we pick the closest one to the predicted set of entities.

    true_summaries = defaultdict(list)
    for obj in datum["objects"]:
        annot_id = obj["precedence"]  # Annotation (set-of-entities) ID
        summary = true_summaries[annot_id]
        summary.append({  # Same format as returned by `Results.summary()`
            'name': categories[obj['category_id']],
            'class': obj['category_id'],
            'confidence': 1.0,
            'box': {
                'x1': obj['bbox'][0],
                'y1': obj['bbox'][1],
                'x2': obj['bbox'][0] + obj['bbox'][2],
                'y2': obj['bbox'][1] + obj['bbox'][3]
            }
        })

    # If we want to visualize the ground truth annotation, create
    # a `Results` instance and invoke its `.show()` method.
    # Otherwise, comment it out.
    """
    for annot_id, summary in true_summaries.items():
        true_results = Results(
            orig_img = np.asarray(datum['image']),
            path = "",
            names = categories,
            boxes = np.array([
                [
                    entry['box']['x1'],
                    entry['box']['y1'],
                    entry['box']['x2'],
                    entry['box']['y2'],
                    entry['confidence'],
                    entry['class']
                ] for entry in summary
            ])
        )
        true_results.show()
    """

    # Perform object detection on the `datum` image
    results = model(datum["image"])[0]

    # Display results
    # results.show()  # Other options: .print(), .save(), .crop(), .pandas(), etc.

    pred_summary = results.summary(normalize = False, decimals = 5)
    # print(pred_summary)

    # If there is more than one human annotation, pick the one with
    # the largest intersection area with the predicted annotation

    largest_intersect_area = -1
    best_true_summary = None
    best_areas = None
    for annot_id, true_summary in true_summaries.items():
        areas = score_summary(pred_summary, true_summary)
        intersect_area = sum(entry['intersect_area'] for entry in areas)
        if intersect_area > largest_intersect_area:
            largest_intersect_area = intersect_area
            best_true_summary = true_summary
            best_areas = areas

    # Update aggregated scores, macro- and micro-

    if best_areas is not None:
        for cid, cname in enumerate(categories):
            aggregated_areas[cid]['pred_area'] += best_areas[cid]['pred_area']
            aggregated_areas[cid]['true_area'] += best_areas[cid]['true_area']
            aggregated_areas[cid]['intersect_area'] += best_areas[cid]['intersect_area']
            if best_areas[cid]['pred_area'] > 0:
                precision = best_areas[cid]['intersect_area'] / best_areas[cid]['pred_area']
                aggregated_micro_scores[cid]['precision'] += precision
                aggregated_micro_counts[cid]['precision'] += 1
            if best_areas[cid]['true_area'] > 0:
                recall = best_areas[cid]['intersect_area'] / best_areas[cid]['true_area']
                aggregated_micro_scores[cid]['recall'] += recall
                aggregated_micro_counts[cid]['recall'] += 1
            if best_areas[cid]['pred_area'] + best_areas[cid]['true_area'] > 0:
                aggregated_micro_scores[cid]['F1'] += (
                    2 * best_areas[cid]['intersect_area'] / (best_areas[cid]['pred_area'] + best_areas[cid]['true_area'])
                    # same as 2 * precision * recall / (precision + recall), but safer
                )
                aggregated_micro_counts[cid]['F1'] += 1

# Print the macro- and micro- precision and recall scores for each category

for cid, cname in enumerate(categories):
    if aggregated_areas[cid]['pred_area'] > 0:
        macro_precision = aggregated_areas[cid]['intersect_area'] / aggregated_areas[cid]['pred_area']
    else:
        macro_precision = None
    if aggregated_areas[cid]['true_area'] > 0:
        macro_recall = aggregated_areas[cid]['intersect_area'] / aggregated_areas[cid]['true_area']
    else:
        macro_recall = None
    if aggregated_areas[cid]['pred_area'] + aggregated_areas[cid]['true_area'] > 0:
        macro_F1 = (
            2 * aggregated_areas[cid]['intersect_area'] / (aggregated_areas[cid]['pred_area'] + aggregated_areas[cid]['true_area'])
            # same as 2 * macro_precision * macro_recall / (macro_precision + macro_recall), but safer
        )
    else:
        macro_F1 = None
    if aggregated_micro_counts[cid]['precision'] > 0:
        micro_precision = aggregated_micro_scores[cid]['precision'] / aggregated_micro_counts[cid]['precision']
    else:
        micro_precision = None
    if aggregated_micro_counts[cid]['recall'] > 0:
        micro_recall = aggregated_micro_scores[cid]['recall'] / aggregated_micro_counts[cid]['recall']
    else:
        micro_recall = None
    if aggregated_micro_counts[cid]['F1'] > 0:
        micro_F1 = aggregated_micro_scores[cid]['F1'] / aggregated_micro_counts[cid]['F1']
    else:
        micro_F1 = None

    print(f"\nCategory {cid}: {cname}")
    print(f"    Macro Precision = {macro_precision}")
    print(f"    Macro Recall = {macro_recall}")
    print(f"    Macro F1 = {macro_F1}")
    print(f"    Micro Precision = {micro_precision}")
    print(f"    Micro Recall = {micro_recall}")
    print(f"    Micro F1 = {micro_F1}")


