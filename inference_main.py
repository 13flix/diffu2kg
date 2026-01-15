"""
Inference script for diffu2kg project.
"""
import os
import json
import torch
import numpy as np
from transformers import set_seed
from model_utils import create_model_and_diffusion
import kg_dataloader
from utils import logger
from evaluation import evaluate_predictions, save_results
from config import Config


def main():
    config = Config()
    
    set_seed(config.seed)
    device = torch.device(config.device)

    logger.configure()

    model, diffusion = create_model_and_diffusion(
        class_cond=config.model.class_cond,
        learn_sigma=config.model.learn_sigma,
        sigma_small=config.model.sigma_small,
        num_channels=config.model.num_channels,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        diffusion_steps=config.model.diffusion_steps,
        noise_schedule=config.model.noise_schedule,
        timestep_respacing="",
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
        load_ckpt=None,
        sequence_len=config.data.max_seq_len_src,
        resume_checkpoint="",
        pad_tok_id=0,
        loss_update_granu=1000,
        schedule_update_stride=1000,
    )

    diffusion._load_time_schedule(config.inference.schedule_path)
    model.load_state_dict(torch.load(config.inference.model_name_or_path, map_location=device))
    model.to(device)
    model.eval()

    val_dataloader = kg_dataloader.get_dataloader(
        config.data.test_path,
        config.data.entity2emb,
        config.data.relation2emb,
        config.data.entity2id,
        config.data.relation2id,
        config.data.query_len,
        config.data.batch_size,
        config.data.max_seq_len,
        config.data.max_seq_len_src,
        config.model.in_channel // 4,
        mode="test"
    )

    num_samples = config.inference.num_samples
    if num_samples <= 0:
        num_samples = len(kg_dataloader.KGDataset(
            config.data.test_path,
            config.data.entity2emb,
            config.data.relation2emb,
            config.data.entity2id,
            config.data.relation2id,
            config.data.query_len,
            "test"
        ))
        logger.log(f"sample count is {num_samples}")
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"the parameter count is {pytorch_total_params}")

    diffusion.rescale_timesteps = True

    logger.log("sampling...")
    logger.log(f"Clamping is set to {config.inference.clamp}")
    
    all_samples = []
    ground_true_samples = []

    while len(all_samples) * config.data.batch_size < num_samples:
        batch, _ = next(val_dataloader)
        model_kwargs = {key: item.to(device) for key, item in batch.items() if 'decoder' not in key}
        sample_shape = (
            config.data.batch_size,
            config.data.max_seq_len_src,
            model.input_transformers.shared.weight.shape[1]
        )
        print('sample_shape', sample_shape)
        
        sample = diffusion.p_sample_loop(
            model,
            sample_shape,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            top_p=-1,
            progress=True,
            tokenizer=None,
            log_verbose=True,
            decoder_inputs=batch['decoder_input_ids'],
            generate_by_q=False,
            generate_by_mix=False,
            generate_by_mix_prob=0,
            generate_by_mix_part=1,
        )

        cands = sample[:, 0, :].squeeze(1)
        gathered_samples = [cands]

        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])
        print('number of sample', len(all_samples), all_samples[0].shape)

        ground_true_samples.extend([batch['decoder_input_ids'].cpu().numpy()])

        logger.log(f"created {len(all_samples) * config.data.batch_size} samples")

    cands = np.concatenate(all_samples, axis=0)
    cands = cands[: num_samples]

    decoded_vectors = []
    for seq in cands:
        decoded_vectors.append(seq)

    ground_true_vectors = []
    ground_true_samples = np.concatenate(ground_true_samples, axis=0)[: num_samples]
    ground_true_samples = ground_true_samples[:, 0, :]
    for seq in ground_true_samples:
        ground_true_vectors.append(seq)

    logger.log("sampling complete")

    save_results(cands, ground_true_samples, config.inference.out_dir)

    metrics = evaluate_predictions(
        decoded_vectors,
        ground_true_vectors,
        config.data.entity2emb,
        config.data.test_path
    )

    print(f"hit@1: {metrics['hit@1']}")
    print(f"hit@3: {metrics['hit@3']}")
    print(f"hit@10: {metrics['hit@10']}")
    print(f"mrr: {metrics['mrr']:.6f}")


if __name__ == "__main__":
    main()
