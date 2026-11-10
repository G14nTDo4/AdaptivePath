# Agentic Visual Reasoning in Whole-Slide Pathology Images via Active Perception

## Overview

AdaptivePath performs whole-slide visual question answering from sparse, multiscale observations selected by its active-perception Navigator. The release contains Navigator training, hierarchical Navigator inference, and the downstream VQA inference pipeline. Given the Navigator output for a slide, the reasoning pipeline:

1. uses a vision-language model as the Morphology Interpreter to describe each selected focus image;
2. aggregates focus descriptions at the region level;
3. uses the Deliberator to compare all answer options and audit the evidence independently at each magnification;
4. uses the Arbiter to integrate evidence audits across magnifications and produce the final WSI-level answer.

The Navigator entry point is `navigator_inference/navigate.py`, which generates multiscale observations directly from WSIs. The reasoning entry point is `vqa_inference.py`, which consumes those observations to produce WSI-level answers.

## 1. Installation

```bash
cd AdaptivePath
conda create -n adaptivepath python=3.10 -y
conda activate adaptivepath

# Select the PyTorch build that matches the local CUDA driver.
pip install torch==2.7.0 torchvision==0.22.0

pip install \
  transformers==4.51.0 \
  accelerate==1.10.0 \
  qwen-vl-utils==0.0.14 \
  sentencepiece==0.2.1 \
  pillow==11.2.1
```

Two pretrained model checkpoints are required:

* `VLM_CHECKPOINT_PATH`: a Qwen2.5-VL checkpoint used by the Morphology Interpreter.
* `LLM_CHECKPOINT_PATH`: a causal language model checkpoint used by the Deliberator and Arbiter.

The script loads the two models on separate GPUs by default.

Navigator training and inference use the packaged dependency lists:

```bash
pip install -r navigator_training/requirements.txt
pip install -r navigator_inference/requirements.txt
```

OpenSlide must also be available at the system level. On Ubuntu, install `libopenslide0` before running Navigator inference.

The reference Navigator experiment used PyTorch 2.7.0, torchvision 0.22.0, NumPy 2.0.2, Pillow 11.2.1, and tensorboardX 2.6.4. The repository includes the minimal customized timm 0.4.12 subset required by the Navigator. Do not install another timm version in its place.

## 2. Navigator Training

The training implementation is located in `navigator_training/`:

```text
navigator_training/
├── main_ppo.py
├── engine_ppo_global_scale_single.py
├── dataset_utils.py
├── optim_factory.py
├── utils.py
├── train_phase1.sh
├── train_phase2.sh
└── models/
    ├── dynamic_deitS_global_scale_single.py
    └── timm/
```

The release contains the scale-adaptive Navigator used by AdaptivePath. Training uses AdamW only and preserves the abnormal-region classification updates, clipped PPO policy updates, and geometric and appearance consistency updates used in the reported experiment.

### 2.1 Abnormal-Region Dataset

Arrange the pathologist-reviewed abnormal-region recognition data as an ImageFolder dataset:

```text
DATA_ROOT/
├── train/
│   ├── abnormal/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── normal/
│       ├── image_001.jpg
│       └── ...
└── val/
    ├── abnormal/
    │   └── ...
    └── normal/
        └── ...
```

The scripts load `train` and `val` through `torchvision.datasets.ImageFolder`. The dataset images are resized to 512 by 512 pixels before the Navigator extracts a 112 by 112 global glance and up to four sequential focus observations.

### 2.2 Phase I: Representation and Policy Learning

Phase I trains the Navigator from scratch:

```bash
bash navigator_training/train_phase1.sh \
  DATA_ROOT \
  OUTPUT_ROOT/navigator_phase1 \
  0
```

The final argument is the CUDA device index and defaults to `0`. During each training epoch, the implementation updates the abnormality representation on every mini-batch and performs clipped PPO updates after every 10 collected mini-batches. The policy predicts two normalized location coordinates and one focus-scale action for each observation. Phase I does not enable the additional focus-consistency objective.

The script reproduces the reference configuration with a per-GPU batch size of 32, a focus budget of at most four steps, an initial learning rate of `1e-4`, 300 epochs, and 10 warmup epochs. Checkpoints are written to the specified output directory. `checkpoint-best.pth` is selected using validation accuracy of the final focus step.

### 2.3 Phase II: Consistency Fine-Tuning

Phase II initializes the same Navigator from the Phase I checkpoint and enables geometric and appearance consistency:

```bash
bash navigator_training/train_phase2.sh \
  DATA_ROOT \
  OUTPUT_ROOT/navigator_phase1/checkpoint-best.pth \
  OUTPUT_ROOT/navigator_phase2 \
  0
```

The reference Phase II configuration uses a learning rate of `5e-5`, 100 epochs, and 5 warmup epochs. Every 10 training steps, an additional policy update aligns focus trajectories under horizontal and vertical flips, 90-degree rotations, and appearance perturbations. The loss includes set-level geometric consistency, step-level geometric consistency, appearance consistency, and focus-scale consistency.

Both scripts use one process and one GPU through `torchrun`. Training resumes automatically when the output directory already contains a compatible checkpoint. The Phase II `checkpoint-best.pth` is the Navigator checkpoint used for downstream multiscale focus extraction.

## 3. Hierarchical Navigator Inference

The WSI inference implementation is located in `navigator_inference/` and reuses the customized model implementation packaged under `navigator_training/models/`:

```text
navigator_inference/
├── navigate.py
├── adaptivepath_navigator.py
├── selection.py
└── requirements.txt
```

Provide the Phase II checkpoint separately. Model weights are not included in the source release. The Navigator accepts a CSV with a required `svs_path` column.

```bash
python navigator_inference/navigate.py \
  --wsi_csv WSI_MANIFEST.csv \
  --ckpt_path OUTPUT_ROOT/navigator_phase2/checkpoint-best.pth \
  --output_root NAVIGATION_ROOT \
  --device cuda:0
```

The Navigator first scores 2.5x candidate regions, uses the retained regions as the 5x search space, and then uses the retained 5x regions as the 10x search space. `K=5` and `T=4` are fixed upper bounds. Tissue and white-background filtering can produce fewer regions or focus images without causing an error. The reasoning pipeline requires at least one usable focus image at the first magnification. It visits a later magnification only when the current evidence review is insufficient and that magnification has a usable focus image.

The command writes `<output_root>/<slide_id>/navigation.json` and the selected focus images under the `2p5x`, `5x`, and `10x` directories. Existing completed slides are skipped unless `--overwrite` is supplied. The Navigator derives `slide_id` from a truncated SHA-256 hash of the source filename, so the filename and source WSI path are not written to `navigation.json` or terminal status messages.

## 4. Reasoning Input Preparation

### 4.1 Navigator Output

Each WSI must have a `navigation.json` file and focus images at 2.5x. Focus images at 5x and 10x are used only if the preceding evidence review requests another round:

```text
NAVIGATION_ROOT/
├── slide_1/
│   ├── navigation.json
│   ├── 2p5x/
│   │   └── focus/
│   │       ├── region_01_focus_00.jpg
│   │       └── ...
│   ├── 5x/
│   │   └── focus/
│   │       └── ...
│   └── 10x/
│       └── focus/
│           └── ...
└── slide_2/
    └── ...
```


The required `navigation.json` structure is:

```json
{
  "slide_id": "slide_1",
  "levels": {
    "2p5x": [
      {
        "region_id": 57,
        "rank": 1,
        "abnormal_score": 2.30,
        "bbox_level0": [0, 0, 4096, 4096],
        "focuses": [
          {
            "step_idx": 0,
            "bbox_level0": [0, 0, 2048, 2048],
            "location_level0": [1024, 1024],
            "image_path": "2p5x/focus/region_01_focus_00.jpg"
          }
        ]
      }
    ],
    "5x": [],
    "10x": []
  }
}
```

The example abbreviates the 5x and 10x entries. A later level may be empty when navigation has no further usable observations. `image_path` must be relative to the directory containing `navigation.json`.

### 4.2 Case Manifest

The simplest input is a JSON manifest that explicitly associates each question with its Navigator output:

```json
[
  {
    "slide_id": "slide_1",
    "question_id": "example_question_01",
    "question": "What type of tumor is the invasive component?",
    "choices": [
      "Ductal adenocarcinoma",
      "Metastatic mammary carcinoma",
      "Tubular formation",
      "Comedo necrosis"
    ],
    "ground_truth": "Ductal adenocarcinoma",
    "navigation_path": "/path/to/slide_1/navigation.json"
  }
]
```

The required fields are `navigation_path`, `question`, and `choices`. `slide_id`, `question_id`, and `ground_truth` are optional. The current pipeline requires multiple-choice questions. For anonymous review, manifests and WSI filenames must use deidentified identifiers and must not contain protected health information.


## 5. Input Validation

Validate the manifest and focus images at the first magnification before loading either model:

```bash
python vqa_inference.py \
  --case_manifest CASE_MANIFEST.json \
  --output_dir RESULTS_DIRECTORY \
  --dry_run
```

The command reports the numbers of selected cases, valid cases, invalid cases, and first-magnification focus images. Validation accepts any positive region count up to `K=5` and zero to four retained focus images per region. Later magnifications are validated only when an insufficient evidence review requests them; an unavailable next magnification ends evidence acquisition. Model checkpoints are not required in dry-run mode.

## 6. Inference

### 6.1 Manifest-Based Inference

```bash
python vqa_inference.py \
  --case_manifest CASE_MANIFEST.json \
  --output_dir RESULTS_DIRECTORY \
  --vlm_ckpt VLM_CHECKPOINT_PATH \
  --llm_ckpt LLM_CHECKPOINT_PATH \
  --cuda_visible_devices 0,1 \
  --vlm_device cuda:0 \
  --llm_device cuda:1
```

`cuda:0` and `cuda:1` refer to the logical device indices after applying `--cuda_visible_devices`.


### 6.2 Sharded Inference

Use the same input with different `--shard_id` values to divide cases across processes:

```bash
python vqa_inference.py \
  --case_manifest CASE_MANIFEST.json \
  --output_dir RESULTS_DIRECTORY \
  --vlm_ckpt VLM_CHECKPOINT_PATH \
  --llm_ckpt LLM_CHECKPOINT_PATH \
  --cuda_visible_devices 0,1 \
  --vlm_device cuda:0 \
  --llm_device cuda:1 \
  --num_shards 4 \
  --shard_id 0
```

Run the command again with shard IDs 1, 2, and 3 on the desired GPU pairs. Each process writes to `RESULTS_DIRECTORY/shard_<shard_id>/`.

## 7. Outputs and Resuming

The output structure is:

```text
RESULTS_DIRECTORY/
├── cache/
│   └── <question_id>/
│       └── ...
└── shard_0/
    ├── <question_id_1>.json
    ├── <question_id_2>.json
    └── ...
```

Each result JSON contains:

* `stages`: focus-level morphology descriptions at 2.5x, 5x, and 10x;
* `level_results`: region aggregation, the independent answer, and the evidence-sufficiency review at each visited magnification;
* `stopping_reason`: sufficient evidence, no next observation, or exhausted magnification budget;
* `final_adjudication`: the Arbiter's final answer and evidence-based reason;
* `pred_answer_text`: the predicted answer text;
* `pred_answer_original_label`: the predicted label in the manifest option order;
* `inference_seconds`: end-to-end runtime for the case.

Partial results are saved after each magnification. A completed case is skipped automatically when the command is restarted. Use `--overwrite` to recompute completed cases. Morphology descriptions remain reusable through `--cache_dir` or the default `RESULTS_DIRECTORY/cache/`.

## 8. Main Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--start_index N` | `0` | Skip the first `N` cases assigned to the current shard |
| `--limit N` | all | Process at most `N` cases after sharding and `start_index` |
| `--vlm_max_new_tokens N` | `512` | Maximum output length for each morphology description |
| `--cache_dir PATH` | `<output_dir>/cache` | Morphology-description cache |
| `--max_memory_gib N` | unset | Per-GPU LLM memory limit when `--llm_device auto` is used |
| `--overwrite` | off | Recompute cases that already contain final adjudication |
| `--dry_run` | off | Validate inputs without loading models |

## 9. Operational Notes

* `abnormal_score` is treated only as Navigator metadata and is not direct diagnostic evidence.
* Each visited magnification is assessed independently. A sufficient review stops observation before the Arbiter integrates the visited evidence.
* Missing or unobserved findings are treated as neutral evidence by the prompts.
* A failed case is saved with `status: "failed"` and only the exception type; local paths and exception text are not persisted.
* If GPU memory is insufficient, use smaller checkpoints, assign the models to different GPUs, reduce `--vlm_max_new_tokens`, or use `--llm_device auto --max_memory_gib N`.
