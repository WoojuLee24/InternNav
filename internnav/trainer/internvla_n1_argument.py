from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    system1: Optional[str] = field(default='nextdit')
    n_query: int = field(default=4)

    bev_mode: str = field(default="none", metadata={"help": "none|occ_gt|rgb_gt|fpv_concat_gt|occ_depthanythingv2|rgb_depthanythingv2"})
    dav2_max_depth: float = field(default=10.0)
    debug_modes: str = field(default="")
    debug_dir: Optional[str] = field(default=None)
    debugpy: str = field(default="", metadata={"help": "debugpy attach target: 'trainer' listens on port 5681 after model load (rank 0 only)"})


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)

    data_root: Optional[str] = field(default=None, metadata={"help": "Root directory prepended to all dataset data_path entries."})
    vln_dataset_use: str = field(default="")
    iign_dataset_use: str = field(default="")
    sample_step: int = field(default=4)
    num_history: Optional[int] = field(default=8)
    predict_step_num: Optional[int] = field(default=32)
    pixel_goal_only: Optional[bool] = field(default=False)
    data_augmentation: Optional[bool] = field(default=False)
    transform_train: Optional[str] = field(default=None)
    resize_h: Optional[int] = field(default=384)
    resize_w: Optional[int] = field(default=384)
    num_future_steps: Optional[int] = field(default=4)
    max_dialog_turns: Optional[int] = field(default=6)
    val_ratio: float = field(default=0.0, metadata={"help": "Fraction of dataset to use for validation. 0.0 disables validation."})
    val_max_samples: int = field(default=0, metadata={"help": "Max number of validation samples. 0 = unlimited."})
    train_max_samples: int = field(default=0, metadata={"help": "Max number of training samples. 0 = unlimited."})
    stop_weight: int = field(default=5, metadata={"help": "Oversampling multiplier for stop samples. Only used when pixel_goal_only=False."})

    # ---- S2 unified image input (independent of S1's bev_*/s1_*; GT depth only) ----
    s2_image_view: str = field(default="fpv", metadata={"help": "fpv|bev"})
    s2_image_type: str = field(default="rgb", metadata={"help": "rgb|depth|panorama (panorama not implemented)"})
    s2_image_mode: str = field(default="raw", metadata={"help": "value-processing mode; vocabulary depends on (view,type)"})
    s2_combine_mode: str = field(default="none", metadata={"help": "none|replace|concat"})
    s2_depth_source: str = field(default="gt", metadata={"help": "gt only for now — estimated depth needs model inference inside dataloader workers, out of scope"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
