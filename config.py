"""
Unified configuration management for diffu2kg project.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = "./ICEWS14s/train.txt"
    val_path: str = "./ICEWS14s/valid.txt"
    test_path: str = "./ICEWS14s/test.txt"
    entity2emb: str = "./ICEWS14s/entityembeddings.json"
    relation2emb: str = "./ICEWS14s/relationembeddings.json"
    entity2id: str = "./ICEWS14s/entity2id.txt"
    relation2id: str = "./ICEWS14s/relation2id.txt"
    query_len: int = 1
    batch_size: int = 32
    max_seq_len: int = 64
    max_seq_len_src: int = 64
    in_out_channel: int = 800


@dataclass
class ModelConfig:
    """Model configuration."""
    num_channels: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    diffusion_steps: int = 4000
    noise_schedule: str = "sqrt"
    timestep_respacing: str = ""
    learn_sigma: bool = False
    sigma_small: bool = False
    class_cond: bool = False
    use_kl: bool = False
    predict_xstart: bool = True
    rescale_timesteps: bool = True
    rescale_learned_sigmas: bool = False
    use_checkpoint: bool = False
    model_arch: str = "bart"
    in_channel: int = 800
    out_channel: int = 800
    training_mode: str = "train"
    vocab_size: int = 10000
    config_name: str = "bart-base-uncased"
    logits_mode: int = 1
    init_pretrained: bool = False
    freeze_embeddings: bool = False
    use_pretrained_embeddings: bool = False
    load_ckpt: Optional[str] = None
    sequence_len: int = 64
    resume_checkpoint: str = "./checkpoints/myexperiment"
    pad_tok_id: int = 0
    loss_update_granu: int = 1000
    schedule_update_stride: int = 1000


@dataclass
class TrainingConfig:
    """Training configuration."""
    lr: float = 1e-4
    ema_rate: float = 0.9999
    log_interval: int = 100
    save_interval: int = 10000
    eval_interval: int = 5000
    weight_decay: float = 0.01
    lr_anneal_steps: int = 500000
    warmup: int = 5000
    gradient_clipping: float = 1.0
    use_fp16: bool = False
    fp16_scale_growth: float = 1e-3
    schedule_sampler: str = "uniform"
    microbatch: int = 32
    resume_checkpoint: str = None


@dataclass
class InferenceConfig:
    """Inference configuration."""
    num_samples: int = 0
    clamp: int = 0
    out_dir: str = "./result"
    schedule_path: str = "./checkpoints/myexperiment/alpha_cumprod_step_151000.npy"
    model_name_or_path: str = "./checkpoints/myexperiment/ema_0.9999_500000.pt"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    checkpoint_path: str = "./checkpoints/myexperiment"
    seed: int = 42
    device: str = "cuda"
    
    def __post_init__(self):
        if self.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
