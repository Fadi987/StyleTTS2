# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
import torch.nn.functional as F
import click
import shutil
import warnings
warnings.simplefilter('ignore')
import wandb

from meldataset import build_dataloader

from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

from accelerate import Accelerator

accelerator = Accelerator()

# simple fix for dataparallel that allows access to class attributes
class AttributeForwardingDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_pretrained_models(config):
    """
    Load all pretrained models required for training.
    
    Args:
        config: Configuration dictionary containing model paths
        device: Device to load the models to
        
    Returns:
        tuple: (text_aligner, pitch_extractor, plbert)
    """
    # load pretrained ASR model
    ASR_config = config['ASR_config']
    ASR_path = config['ASR_path']
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config['F0_path']
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config['PLBERT_dir']
    plbert = load_plbert(BERT_path)
    
    return text_aligner, pitch_extractor, plbert

def setup_optimizers(model, optimizer_params, epochs, train_dataloader_length):
    """
    Set up optimizers with appropriate learning rates and parameters for different model components.
    
    Args:
        model: Dictionary containing model components
        optimizer_params: Parameters for optimizer configuration
        epochs: Total number of training epochs
        train_dataloader_length: Length of the training dataloader
        
    Returns:
        optimizer: Configured optimizer object
    """
    # Base scheduler parameters
    base_scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": train_dataloader_length,
    }

    # Create scheduler params dictionary for each model component
    scheduler_params_dict = {key: base_scheduler_params.copy() for key in model}
    
    # Set specific learning rates for certain components
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    # Build optimizer with model parameters and scheduler parameters
    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict=scheduler_params_dict, 
        lr=optimizer_params.lr
    )
    
    # Configure BERT-specific optimizer parameters
    for param_group in optimizer.optimizers['bert'].param_groups:
        param_group['betas'] = (0.9, 0.99)
        param_group['lr'] = optimizer_params.bert_lr
        param_group['initial_lr'] = optimizer_params.bert_lr
        param_group['min_lr'] = 0
        param_group['weight_decay'] = 0.01
        
    # Configure acoustic module optimizer parameters
    acoustic_modules = ["decoder", "style_encoder"]
    for module_name in acoustic_modules:
        for param_group in optimizer.optimizers[module_name].param_groups:
            param_group['betas'] = (0.0, 0.99)
            param_group['lr'] = optimizer_params.ft_lr
            param_group['initial_lr'] = optimizer_params.ft_lr
            param_group['min_lr'] = 0
            param_group['weight_decay'] = 1e-4
    
    return optimizer

def perform_text_alignment(model, mels, mask, texts):
    """
    Perform text alignment using the text aligner model.
    
    Args:
        model: Model containing the text_aligner component
        mels: Mel spectrograms
        mask: Mask for the mel spectrograms
        texts: Text inputs
        
    Returns:
        tuple: (alignment_logits, alignment_attn) - speech-to-speech predictions and attention
        
    Raises:
        Exception: If text alignment fails
    """
    _, alignment_logits, alignment_attn = model.text_aligner(mels, mask, texts)
    alignment_attn = alignment_attn.transpose(-1, -2)
    alignment_attn = alignment_attn[..., 1:]
    alignment_attn = alignment_attn.transpose(-1, -2)
    
    return alignment_logits, alignment_attn

def calculate_duration_and_ce_losses(duration_predictions, duration_targets, input_lengths):
    """
    Calculate duration and cross-entropy losses for text-to-speech alignment.
    
    Args:
        duration_predictions: Predicted durations from the model
        duration_targets: Ground truth duration targets
        input_lengths: Lengths of the input sequences
        
    Returns:
        tuple: (loss_dur_ce, loss_dur) - Cross-entropy loss and duration loss
    """
    loss_dur_ce = 0
    loss_dur = 0
    batch_size = len(duration_predictions)
    
    for pred, target, length in zip(duration_predictions, duration_targets, input_lengths):
        # Trim predictions to actual sequence length
        pred = pred[:length, :]
        target = target[:length].long()
        
        # Create binary target matrix
        binary_target = torch.zeros_like(pred)
        for i in range(binary_target.shape[0]):
            binary_target[i, :target[i]] = 1
        
        # Calculate duration prediction by applying sigmoid and summing
        dur_pred = torch.sigmoid(pred).sum(axis=1)
        
        # Skip first and last tokens for duration loss (typically BOS/EOS tokens)
        loss_dur += F.l1_loss(dur_pred[1:length-1], target[1:length-1])
        
        # Calculate cross-entropy loss on flattened predictions and targets
        loss_dur_ce += F.binary_cross_entropy_with_logits(pred.flatten(), binary_target.flatten())
    
    # Normalize losses by batch size
    loss_dur_ce /= batch_size
    loss_dur /= batch_size
    
    return loss_dur_ce, loss_dur

def create_random_segments(mel_input_length, aligned_encoded_text, predictor_features, mels, waves, device, max_len=None):
    """
    Create random segments from the input data for training.
    
    Args:
        mel_input_length: Tensor containing the lengths of mel spectrograms
        aligned_encoded_text: Tensor containing ASR features
        predictor_features: Tensor containing predictor outputs
        mels: Tensor containing mel spectrograms
        waves: List of audio waveforms
        device: Device to place tensors on
        max_len: Optional maximum length constraint
        
    Returns:
        tuple: (encoder_features, predictor_features, mel_targets, waveforms)
            - encoder_features: Encoder features segments
            - predictor_features: Predictor features segments
            - mel_targets: Ground truth mel segments
            - waveforms: Audio waveform segments
    """
    # Calculate segment lengths
    mel_len_content = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2 if max_len else float('inf'))
    
    # Initialize lists for collecting segments
    encoder_segments = []
    predictor_segments = []
    mel_targets = []
    waveforms = []
    
    # Process each item in the batch
    for batch_idx in range(len(mel_input_length)):
        mel_length = int(mel_input_length[batch_idx].item() / 2)
        
        # Create content segments with consistent random start point
        content_start = np.random.randint(0, mel_length - mel_len_content)
        
        # Extract encoder features
        encoder_segments.append(aligned_encoded_text[batch_idx, :, content_start:content_start+mel_len_content])
        
        # Extract predictor features
        predictor_segments.append(predictor_features[batch_idx, :, content_start:content_start+mel_len_content])
        
        # Extract ground truth mel segments (at 2x resolution)
        mel_targets.append(mels[batch_idx, :, (content_start * 2):((content_start+mel_len_content) * 2)])
        
        # Extract corresponding audio waveform segments
        # Note: 300 is the hop length ratio between audio and mel
        audio_start = (content_start * 2) * 300
        audio_end = ((content_start+mel_len_content) * 2) * 300
        waveform = waves[batch_idx][audio_start:audio_end]
        waveforms.append(torch.from_numpy(waveform).to(device))
        
    # Stack all segments into tensors
    return (
        torch.stack(encoder_segments),
        torch.stack(predictor_segments),
        torch.stack(mel_targets).unsqueeze(1).detach(),
        torch.stack(waveforms).unsqueeze(1).float().detach(),
    )

def compute_diffusion_loss(model, target_style, bert_embeddings, sampler, multispeaker=False, reference_features=None, device=None, diffusion_steps_range=(3, 5)):
    """
    Compute the diffusion loss.
    
    Args:
        model: Dictionary containing model components
        target_style: Target style tensor to be predicted by the diffusion model
        bert_embeddings: BERT duration embeddings used as conditioning
        multispeaker: Whether the model supports multiple speakers
        reference_features: Reference features for multispeaker models
        device: Device to place tensors on
        diffusion_steps_range: Range for random number of diffusion steps (min, max)
        
    Returns:
        tuple: (diffusion_loss, style_recon_loss, style_predictions, estimated_sigma)
            - diffusion_loss: Diffusion loss (EDM)
            - style_recon_loss: Style reconstruction loss
            - style_predictions: Style predictions from the diffusion model
            - estimated_sigma: Estimated sigma data (if applicable)
    """
    # Sample random number of diffusion steps
    num_diffusion_steps = np.random.randint(*diffusion_steps_range)
    
    # Estimate sigma data if configured
    diffusion_model = model.diffusion.module.diffusion
    
    # Create input noise with same shape as target
    input_noise = torch.randn_like(target_style).unsqueeze(1).to(device)
    
    # Generate predictions based on whether it's multispeaker
    if multispeaker:
        # Generate style predictions using the sampler
        style_predictions = sampler(
            noise=input_noise,
            embedding=bert_embeddings,
            embedding_scale=1,
            features=reference_features,  # reference from the same speaker
            embedding_mask_proba=0.1,
            num_steps=num_diffusion_steps
        ).squeeze(1)
        
        # Calculate EDM loss with reference features
        diffusion_loss = model.diffusion(
            target_style.unsqueeze(1), 
            embedding=bert_embeddings, 
            features=reference_features
        ).mean()
    else:
        # Generate style predictions using the sampler
        style_predictions = sampler(
            noise=input_noise,
            embedding=bert_embeddings,
            embedding_scale=1,
            embedding_mask_proba=0.1,
            num_steps=num_diffusion_steps
        ).squeeze(1)
        
        # Calculate EDM loss without reference features
        diffusion_loss = diffusion_model(
            target_style.unsqueeze(1), 
            embedding=bert_embeddings
        ).mean()
    
    # Style reconstruction loss between predictions and target
    style_recon_loss = F.l1_loss(style_predictions, target_style)
    
    return diffusion_loss, style_recon_loss 

def extract_style_features(model, mels, mel_input_lengths):
    """
    Extract prosodic and acoustic style features from mel spectrograms.
    
    This operation is done per-utterance because of the avgpool layer.
    
    Args:
        model: Dictionary containing model components
        mels: Tensor containing mel spectrograms [batch_size, n_mels, time]
        mel_input_lengths: Tensor containing the lengths of mel spectrograms
        
    Returns:
        tuple: (prosodic_styles, acoustic_styles)
            - prosodic_styles: Global prosodic style features
            - acoustic_styles: Global acoustic style features
    """
    prosodic_features = []
    acoustic_features = []
    
    # Process each utterance in the batch individually
    for batch_idx in range(len(mel_input_lengths)):
        # Extract mel up to its actual length
        mel = mels[batch_idx, :, :mel_input_lengths[batch_idx]]
        
        # Reshape to add batch and channel dimensions [1, 1, n_mels, time]
        mel_expanded = mel.reshape(1, 1, *mel.shape)
        
        # Extract prosodic style features
        prosodic = model.predictor_encoder(mel_expanded)
        prosodic_features.append(prosodic)
        
        # Extract acoustic style features
        acoustic = model.style_encoder(mel_expanded)
        acoustic_features.append(acoustic)
    
    # Stack features from all utterances in batch
    prosodic_features = torch.stack(prosodic_features)
    acoustic_features = torch.stack(acoustic_features)
    
    # Remove extra dimensions (keeping batch dimension)
    prosodic_features = prosodic_features.squeeze(tuple(range(1, len(prosodic_features.shape))))
    acoustic_features = acoustic_features.squeeze(tuple(range(1, len(acoustic_features.shape))))
    
    return prosodic_features, acoustic_features

@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    
    # Initialize wandb
    wandb.init(project="test", config=config)

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)
    
    batch_size = config['batch_size']

    epochs = config['epochs']
    save_freq = config['save_freq']
    log_interval = config['log_interval']

    data_params = config['data_params']
    sr = config['preprocess_params']['sr']
    train_path = data_params['train_data']
    val_path = data_params['val_data']

    max_len = config['max_len']
    
    loss_params = Munch(config['loss_params'])
    diffusion_training_epoch = loss_params.diffusion_training_epoch
    joint_training_epoch = loss_params.joint_training_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = accelerator.device

    train_dataloader = build_dataloader(train_list, batch_size=batch_size, num_workers=2, device=device, **data_params)
    val_dataloader = build_dataloader(val_list, validation=True, batch_size=batch_size, num_workers=0, device=device, **data_params)
    
    # Load pretrained models
    text_aligner, pitch_extractor, plbert = load_pretrained_models(config)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = AttributeForwardingDataParallel(model[key])
            
    generator_adv_loss = AttributeForwardingDataParallel(GeneratorLoss(model.mpd, model.msd).to(device))
    discriminator_adv_loss = AttributeForwardingDataParallel(DiscriminatorLoss(model.mpd, model.msd).to(device))
    wav_lm_loss = AttributeForwardingDataParallel(WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device))

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    optimizer = setup_optimizers(
        model, 
        optimizer_params, 
        epochs, 
        len(train_dataloader)
    )
    
    # load models if there is a model
    start_epoch = 0
    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True), ignore_modules=['bert'])

    n_down = model.text_aligner.n_down

    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    running_std = []

    slmadv_params = config['slmadv_params']
    slmadv = SLMAdversarialLoss(model, wav_lm_loss, sampler, **slmadv_params)
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    best_loss = float('inf'); iters = 0

    torch.cuda.empty_cache()
    for epoch in range(start_epoch, epochs):
        running_loss = 0

        _ = [model[key].eval() for key in model]
        _ = [model[key].train() for key in ['text_aligner', 'text_encoder', 'predictor', 'bert_encoder', 'bert', 'msd', 'mpd']]

        for batch_idx, batch in enumerate(train_dataloader):
            waves, texts, bert_texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
            _ = [b.to(device) for b in batch[1:]]

            if mels.size(-1) < 80:
                continue

            mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            try:
                alignment_logits, alignment_attn = perform_text_alignment(model, mels, mask, texts)
            except:
                continue

            loss_algn_ce = 0
            for _alignment_logits, _text_input, _text_length in zip(alignment_logits, texts, input_lengths):
                loss_algn_ce += F.cross_entropy(_alignment_logits[:_text_length], _text_input[:_text_length])
            loss_algn_ce /= texts.size(0)
            
            mask_ST = mask_from_lens(alignment_attn, input_lengths, mel_input_length // (2 ** n_down))
            alignment_attn_mono = maximum_path(alignment_attn, mask_ST)
            loss_algn_mono = F.l1_loss(alignment_attn, alignment_attn_mono) * 10

            # encode
            text_encoded = model.text_encoder(texts, input_lengths, text_mask)
            
            # Randomly choose between regular and monotonic attention for alignment
            attn_matrix = alignment_attn_mono if random.random() > 0.5 else alignment_attn
            aligned_encoded_text = text_encoded @ attn_matrix
            duration_ground_truth = alignment_attn_mono.sum(axis=-1).detach()

            # Extract style features for the entire utterance
            utterance_prosodic_style, utterance_acoustic_style = extract_style_features(model, mels, mel_input_length)
            
            # Combine features for denoiser ground truth
            target_style = torch.cat([utterance_acoustic_style, utterance_prosodic_style], dim=-1).detach()

            bert_embeddings = model.bert(bert_texts, attention_mask=(~text_mask).int())
            bert_encoded = model.bert_encoder(bert_embeddings).transpose(-1, -2) 

            with torch.no_grad():
                # compute reference styles
                if multispeaker and epoch >= diffusion_training_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)
            
            # denoiser training
            if epoch >= diffusion_training_epoch:
                diffusion_loss, style_recon_loss = compute_diffusion_loss(
                    model=model,
                    target_style=target_style,
                    bert_embeddings=bert_embeddings,
                    sampler=sampler,
                    multispeaker=multispeaker,
                    reference_features=ref if multispeaker else None,
                    device=device
                )

                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = target_style.std(axis=-1).mean().item() # batch-wise std estimation
                    running_std.append(model.diffusion.module.diffusion.sigma_data)
                    
                loss_diff_edm = diffusion_loss
                loss_diff_l1 = style_recon_loss
            else:
                loss_diff_l1 = 0
                loss_diff_edm = 0

                
            duration_pred, predictor_features = model.predictor(bert_encoded, utterance_prosodic_style, input_lengths, alignment_attn_mono, text_mask)
            loss_dur_ce, loss_dur = calculate_duration_and_ce_losses(duration_pred, duration_ground_truth, input_lengths)
                
            # Create random segments for training
            encoder_segments, predictor_segments, mel_targets_segments, waveforms_segments = create_random_segments(
                mel_input_length=mel_input_length,
                aligned_encoded_text=aligned_encoded_text,
                predictor_features=predictor_features,
                mels=mels,
                waves=waves,
                device=device,
                max_len=max_len
            )
            
            if mel_targets_segments.size(-1) < 80:
                continue
            
            # Extract style features from the random segments
            segment_acoustic_style = model.style_encoder(mel_targets_segments)           
            segment_prosodic_style = model.predictor_encoder(mel_targets_segments)
                
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(mel_targets_segments)
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                N_real = log_norm(mel_targets_segments).squeeze(1)
                decoder_pred_prosody_real = model.decoder(encoder_segments, F0_real, N_real, segment_acoustic_style)

            F0_fake, N_fake = model.predictor.F0Ntrain(predictor_segments, segment_prosodic_style)
            loss_F0 =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm = F.smooth_l1_loss(N_real, N_fake)

            decoder_pred_prosody_pred = model.decoder(encoder_segments, F0_fake, N_fake, segment_acoustic_style)

            optimizer.zero_grad()
            d_loss = discriminator_adv_loss(waveforms_segments, decoder_pred_prosody_pred.detach()).mean()
            accelerator.backward(d_loss)
            optimizer.step('msd')
            optimizer.step('mpd')

            # generator loss
            optimizer.zero_grad()

            loss_mel = stft_loss(decoder_pred_prosody_pred, waveforms_segments)
            loss_gen_adv = generator_adv_loss(waveforms_segments, decoder_pred_prosody_pred).mean()
            loss_gen_wavlm_adv = wav_lm_loss(waveforms_segments.squeeze(tuple(range(1, len(waveforms_segments.shape)))), decoder_pred_prosody_pred.squeeze(tuple(range(1, len(decoder_pred_prosody_pred.shape))))).mean()

            g_loss = loss_params.lambda_mel * loss_mel + \
                     loss_params.lambda_F0 * loss_F0 + \
                     loss_params.lambda_ce * loss_dur_ce + \
                     loss_params.lambda_norm * loss_norm + \
                     loss_params.lambda_dur * loss_dur + \
                     loss_params.lambda_gen * loss_gen_adv + \
                     loss_params.lambda_slm * loss_gen_wavlm_adv + \
                     loss_params.lambda_sty * loss_diff_l1 + \
                     loss_params.lambda_diff * loss_diff_edm + \
                    loss_params.lambda_mono * loss_algn_mono + \
                    loss_params.lambda_s2s * loss_algn_ce
            
            running_loss += loss_mel.item()
            accelerator.backward(g_loss)

            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('predictor_encoder')
            optimizer.step('style_encoder')
            optimizer.step('decoder')
            
            optimizer.step('text_encoder')
            optimizer.step('text_aligner')
            
            if epoch >= diffusion_training_epoch:
                optimizer.step('diffusion')

            d_loss_slm, loss_gen_lm = 0, 0
            if epoch >= joint_training_epoch:
                # randomly pick whether to use in-distribution text
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts
                    
                slm_out = slmadv(batch_idx, 
                                 waveforms_segments, 
                                 decoder_pred_prosody_real, 
                                 waves, 
                                 mel_input_length,
                                 ref_texts, 
                                 ref_lengths, use_ind, target_style, ref if multispeaker else None)

                if slm_out is not None:
                    d_loss_slm, loss_gen_lm, _ = slm_out

                    # SLM generator loss
                    optimizer.zero_grad()
                    accelerator.backward(loss_gen_lm)

                    # compute the gradient norm
                    total_norm = {}
                    for key in model.keys():
                        total_norm[key] = 0
                        parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    # gradient scaling
                    gradient_scaling = slmadv_params['scale']
                    if total_norm['predictor'] > slmadv_params['thresh']:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= (1 / total_norm['predictor'])

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= gradient_scaling

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= gradient_scaling

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= gradient_scaling
                    
                    optimizer.step('bert_encoder')
                    optimizer.step('bert')
                    optimizer.step('predictor')
                    optimizer.step('diffusion')

                    # SLM discriminator loss
                    if d_loss_slm != 0:
                        optimizer.zero_grad()
                        accelerator.backward(d_loss_slm)
                        optimizer.step('wd')

            iters = iters + 1
            
            if (batch_idx+1)%log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, LM Loss: %.5f, Gen Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f, DiscLM Loss: %.5f, GenLM Loss: %.5f, S2S Loss: %.5f, Mono Loss: %.5f'
                    %(epoch+1, epochs, batch_idx+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_dur, loss_dur_ce, loss_norm, loss_F0, loss_gen_wavlm_adv, loss_gen_adv, loss_diff_l1, loss_diff_edm, d_loss_slm, loss_gen_lm, loss_algn_ce, loss_algn_mono))
                
                # Log metrics to wandb
                wandb.log({
                    'train/Mel Reconstruction Loss': running_loss / log_interval,
                    'train/Generator Adversarial Loss': loss_gen_adv,
                    'train/Discriminator Adversarial Loss': d_loss,
                    'train/Duration Cross-Entropy Loss': loss_dur_ce,
                    'train/Duration Loss': loss_dur,
                    'train/WavLM Embedding Similarity Adversarial Loss': loss_gen_wavlm_adv,
                    'train/Energy Loss': loss_norm,
                    'train/F0 Loss': loss_F0,
                    'train/Diffusion L1 Reconstruction Loss': loss_diff_l1,
                    'train/Diffusion EDM Loss': loss_diff_edm,
                    'train/WavLM Discriminator Adversarial Loss': d_loss_slm,
                    'train/WavLM Generator Adversarial Loss': loss_gen_lm,
                    'train/Alignment Cross-Entropy Loss': loss_algn_ce,
                    'train/Alignment Monotonic Attention Loss': loss_algn_mono,
                    'iteration': iters
                })
                
                running_loss = 0
                
        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for _, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves, texts, bert_texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    _ = [b.to(device) for b in batch[1:]]

                    if mels.size(-1) < 80:
                        continue

                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    text_mask = length_to_mask(input_lengths).to(device)

                    alignment_logits, alignment_attn = perform_text_alignment(model, mels, mask, texts)

                    mask_ST = mask_from_lens(alignment_attn, input_lengths, mel_input_length // (2 ** n_down))
                    alignment_attn_mono = maximum_path(alignment_attn, mask_ST)

                    # encode
                    text_encoded = model.text_encoder(texts, input_lengths, text_mask)
                    aligned_encoded_text = (text_encoded @ alignment_attn_mono)

                    duration_ground_truth = alignment_attn_mono.sum(axis=-1).detach()

                    # Extract style features for validation
                    utterance_prosodic_style, _ = extract_style_features(model, mels, mel_input_length)
                    
                    bert_embeddings = model.bert(bert_texts, attention_mask=(~text_mask).int())
                    bert_encoded = model.bert_encoder(bert_embeddings).transpose(-1, -2) 
                    duration_pred, predictor_features = model.predictor(bert_encoded, utterance_prosodic_style, input_lengths, alignment_attn_mono, text_mask)

                    # Create random segments for training
                    encoder_segments, predictor_segments, mel_targets_segments, waveforms_segments = create_random_segments(
                        mel_input_length=mel_input_length,
                        aligned_encoded_text=aligned_encoded_text,
                        predictor_features=predictor_features,
                        mels=mels,
                        waves=waves,
                        device=device,
                        max_len=max_len
                    )
                    
                    if mel_targets_segments.size(-1) < 80:
                        continue

                    segment_prosodic_style = model.predictor_encoder(mel_targets_segments)
                    F0_fake, N_fake = model.predictor.F0Ntrain(predictor_segments, segment_prosodic_style)
                    _, loss_dur = calculate_duration_and_ce_losses(duration_pred, duration_ground_truth, input_lengths)
                    
                    segment_acoustic_style = model.style_encoder(mel_targets_segments)
                    decoder_pred_prosody_pred = model.decoder(encoder_segments, F0_fake, N_fake, segment_acoustic_style)
                    loss_mel = stft_loss(decoder_pred_prosody_pred.squeeze(), waveforms_segments)

                    F0_real, _, F0 = model.pitch_extractor(mel_targets_segments) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except:
                    continue
        
        # Log metrics to wandb
        wandb.log({
            "eval/Mel Reconstruction Loss": loss_test / iters_test,
            "eval/Duration Loss": loss_align / iters_test,
            "eval/F0 Loss": loss_f / iters_test,
            "epoch": epoch + 1
        })

        if (epoch + 1) % save_freq == 0 :
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            print(f'Saving to {save_path}')
            torch.save(state, save_path)

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

                            
if __name__=="__main__":
    main()
