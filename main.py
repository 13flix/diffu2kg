import torch
from model_utils import create_model_and_diffusion
import kg_dataloader
from transformers import set_seed
from diffusion.resample import create_named_schedule_sampler
from trainer import Trainer
from utils import logger
import os
import pathlib
from config import Config

def main():
    config = Config()
    set_seed(config.seed)
    device = torch.device(config.device)

    checkpoint_path = config.checkpoint_path

    logger.configure(dir=os.path.join(checkpoint_path, 'logger/'))

    logger.log("creating data loader")
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    train_loader = kg_dataloader.get_dataloader(
        config.data.train_path,
        config.data.entity2emb,
        config.data.relation2emb,
        config.data.entity2id,
        config.data.relation2id,
        config.data.query_len,
        config.data.batch_size,
        config.data.max_seq_len,
        config.data.max_seq_len_src,
        config.data.in_out_channel//4,
        mode="train"
    )

    val_loader = kg_dataloader.get_dataloader(
        config.data.val_path,
        config.data.entity2emb,
        config.data.relation2emb,
        config.data.entity2id,
        config.data.relation2id,
        config.data.query_len,
        config.data.batch_size,
        config.data.max_seq_len,
        config.data.max_seq_len_src,
        config.data.in_out_channel//4,
        mode="test"
    )

    logger.log("creating model and diffusion...", checkpoint_path)
    model, diffusion = create_model_and_diffusion(
        class_cond=config.model.class_cond,
        learn_sigma=config.model.learn_sigma,
        sigma_small=config.model.sigma_small,
        num_channels=config.model.num_channels,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        diffusion_steps=config.model.diffusion_steps,
        noise_schedule=config.model.noise_schedule,
        timestep_respacing=config.model.timestep_respacing,
        use_kl=config.model.use_kl,
        predict_xstart=config.model.predict_xstart,
        rescale_timesteps=config.model.rescale_timesteps,
        rescale_learned_sigmas=config.model.rescale_learned_sigmas,
        use_checkpoint=config.model.use_checkpoint,
        model_arch=config.model.model_arch,
        in_channel=config.model.in_channel,
        out_channel=config.model.out_channel,
        training_mode=config.model.training_mode,
        vocab_size=config.model.vocab_size,
        config_name=config.model.config_name,
        logits_mode=config.model.logits_mode,
        init_pretrained=config.model.init_pretrained,
        freeze_embeddings=config.model.freeze_embeddings,
        use_pretrained_embeddings=config.model.use_pretrained_embeddings,
        load_ckpt=config.model.load_ckpt,
        sequence_len=config.model.sequence_len,
        resume_checkpoint=config.model.resume_checkpoint,
        pad_tok_id=config.model.pad_tok_id,
        loss_update_granu=config.model.loss_update_granu,
        schedule_update_stride=config.model.schedule_update_stride,
    )
    model.to(device)

    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f"the parameter count is {pytorch_total_params}")

    schedule_sampler = config.training.schedule_sampler
    schedule_sampler = create_named_schedule_sampler(schedule_sampler, diffusion)

    logger.log(f"saving the hyperparameters to {checkpoint_path}/training_args.json")

    logger.log("training...")
    Trainer(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=config.data.batch_size,
        microbatch=config.training.microbatch,
        lr=config.training.lr,
        ema_rate=config.training.ema_rate,
        log_interval=config.training.log_interval,
        save_interval=config.training.save_interval,
        resume_checkpoint=config.training.resume_checkpoint,
        use_fp16=config.training.use_fp16,
        fp16_scale_growth=config.training.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=config.training.weight_decay,
        lr_anneal_steps=config.training.lr_anneal_steps,
        checkpoint_path=checkpoint_path,
        gradient_clipping=config.training.gradient_clipping,
        eval_data=val_loader,
        eval_interval=config.training.eval_interval,
        warmup=config.training.warmup,
    ).run_loop()

if __name__ == "__main__":
    main()
