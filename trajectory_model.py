import os
import pickle
import time 
from typing import Dict, Tuple, List, Callable, Iterator, Optional, Any, NamedTuple
import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import time
import logging
import onnxruntime as ort
from functools import partial
#from utils.trajectory_utils import ModelConfig, TrajectorySequence
from trajectory_utils import ModelConfig, TrajectorySequence, write_debug_images
from preprocess_dataset import create_memmap_batch_provider

# Configure logging
logger = logging.getLogger("trajectory_model")
    
class FrameEncoder(nn.Module):
    """CNN encoder that returns intermediate feature maps for skip connections."""
    out_channels: Tuple[int, int, int] = (32, 64, 64)
    
    @nn.compact
    def __call__(self, x, *, training: bool):
        features = []
        
        # First convolution: 320→80 (stride 4)
        x = nn.Conv(self.out_channels[0], (8, 8), (4, 4))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        features.append(x)  # First skip connection feature
        
        # Second convolution: 80→40 (stride 2)
        x = nn.Conv(self.out_channels[1], (4, 4), (2, 2))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        features.append(x)  # Second skip connection feature
        
        # Third convolution: 40→40 (stride 1)
        x = nn.Conv(self.out_channels[2], (3, 3), (1, 1))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        features.append(x)  # Final feature map
        
        return x  # Return only the final features for backward compatibility
    
    def apply_intermediate(self, variables, x, *, training: bool):
        """Apply the model and return intermediate feature maps."""
        features = []
        
        # Apply each layer and collect outputs
        params = variables['params']
        
        # First convolution: 320→80 (stride 4)
        x = nn.Conv(self.out_channels[0], (8, 8), (4, 4)).apply({'params': params['Conv_0']}, x)
        x = nn.LayerNorm().apply({'params': params['LayerNorm_0']}, x)
        x = nn.relu(x)
        features.append(x)  # Scale 1/4
        
        # Second convolution: 80→40 (stride 2)
        x = nn.Conv(self.out_channels[1], (4, 4), (2, 2)).apply({'params': params['Conv_1']}, x)
        x = nn.LayerNorm().apply({'params': params['LayerNorm_1']}, x)
        x = nn.relu(x)
        features.append(x)  # Scale 1/8
        
        # Third convolution: 40→40 (stride 1)
        x = nn.Conv(self.out_channels[2], (3, 3), (1, 1)).apply({'params': params['Conv_2']}, x)
        x = nn.LayerNorm().apply({'params': params['LayerNorm_2']}, x)
        x = nn.relu(x)
        features.append(x)  # Scale 1/8 (refined)
        
        return features


class FrameEncoder_old(nn.Module):
    """Shared CNN applied to RGB or mask frames."""
    out_channels: Tuple[int, int, int] = (32, 64, 64)

    @nn.compact
    def __call__(self, x, *, training: bool):
        for i, ch in enumerate(self.out_channels):
            x = nn.Conv(ch, (8, 8) if i == 0 else (4, 4) if i == 1 else (3, 3),
                        (4, 4) if i == 0 else (2, 2) if i == 1 else (1, 1))(x)
            x = nn.LayerNorm()(x)  # Replace BatchNorm with LayerNorm
            x = nn.relu(x)
        return x


def np_to_jax_batch(
    rgb_batch: np.ndarray,
    mask_batch: np.ndarray, 
    target_batch: np.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert NumPy arrays to JAX arrays.
    
    Args:
        rgb_batch: NumPy array of RGB frames
        mask_batch: NumPy array of mask frames
        target_batch: NumPy array of target positions
        
    Returns:
        Tuple of JAX arrays
    """
    return (
        jnp.array(rgb_batch),
        jnp.array(mask_batch),
        jnp.array(target_batch)
    )
    

class SpatiotemporalAttention(nn.Module):
    config: ModelConfig
    
    @nn.compact
    def __call__(self, rgb_frames, mask_frames, *, training=False):
        B, T, H, W, _ = rgb_frames.shape
        
        # Original shared encoders
        rgb_encoder = FrameEncoder(name="rgb_enc")
        mask_encoder = FrameEncoder(out_channels=(16, 32, 32), name="mask_enc")
        
        # Define functions with fixed training parameter
        def encode_rgb(frame):
            return rgb_encoder(frame, training=training)
        
        def encode_mask(frame):
            return mask_encoder(frame, training=training)
        
        # Apply encoders to each frame to get final features
        rgb_feats = jax.vmap(encode_rgb, in_axes=1, out_axes=1)(rgb_frames)  # [B, T, H', W', C1]
        mask_feats = jax.vmap(encode_mask, in_axes=1, out_axes=1)(mask_frames)  # [B, T, H', W', C2]
        
        # Concatenate features along the channel dimension
        feats = jnp.concatenate([rgb_feats, mask_feats], axis=-1)  # [B, T, H', W', C]
        
        # Get dimensions after convolutions
        B, T, H_enc, W_enc, C = feats.shape
        
        # Reshape to tokens: flatten spatial dimensions to tokens
        feats = feats.reshape(B, T * H_enc * W_enc, C)  # [B, T*H'*W', C]
        
        # Project to embedding dimension
        feats = nn.Dense(self.config.embedding_dim)(feats)
        feats = nn.LayerNorm()(feats)
        
        # Add positional encoding
        pos = self.param("pos_embedding",
                        nn.initializers.normal(0.02),
                        (1, T * H_enc * W_enc, self.config.embedding_dim))
        feats = feats + pos
        
        # Self-attention over the tokens
        attn_output = nn.SelfAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.embedding_dim,
            dropout_rate=self.config.dropout_rate
        )(feats, deterministic=not training)
        
        feats = nn.LayerNorm()(feats + attn_output)
        
        # Project features before decoding
        feats = nn.Dense(self.config.embedding_dim)(feats)
        feats = nn.relu(feats)
        feats = nn.Dropout(self.config.dropout_rate)(feats, deterministic=not training)
        
        # Reshape back to spatial representation
        feats = feats.reshape(B, T, H_enc, W_enc, self.config.embedding_dim)
        
        # Pool over time dimension - mean pooling
        feats = feats.mean(axis=1)  # [B, H_enc, W_enc, embedding_dim]
        
        # ------ Create skip connections ------
        # We need to manually create intermediate features at different scales
        # These are the same scales that would be produced by the encoder
        
        # For RGB: Process at 80x80 and 40x40 scales
        skip_80_rgb = jax.vmap(
            lambda frame: nn.Conv(self.config.feature_dim, (8, 8), (4, 4))(frame), 
            in_axes=1, out_axes=1
        )(rgb_frames)
        skip_80_rgb = jax.vmap(lambda x: nn.LayerNorm()(x), in_axes=1, out_axes=1)(skip_80_rgb)
        skip_80_rgb = jax.vmap(lambda x: nn.relu(x), in_axes=1, out_axes=1)(skip_80_rgb)
        
        # For masks: Process at 80x80 scale
        skip_80_mask = jax.vmap(
            lambda frame: nn.Conv(16, (8, 8), (4, 4))(frame), 
            in_axes=1, out_axes=1
        )(mask_frames)
        skip_80_mask = jax.vmap(lambda x: nn.LayerNorm()(x), in_axes=1, out_axes=1)(skip_80_mask)
        skip_80_mask = jax.vmap(lambda x: nn.relu(x), in_axes=1, out_axes=1)(skip_80_mask)
        
        # Combine and pool across time
        skip_80 = jnp.concatenate([skip_80_rgb, skip_80_mask], axis=-1).mean(axis=1)
        
        # ------ U-Net decoder with skip connections ------
        x = feats  # Start from bottleneck [B, 40, 40, embedding_dim]
        C = self.config.embedding_dim
        
        # First up-block: 40→80
        x = nn.ConvTranspose(C//2, (4, 4), (2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Concatenate with 80x80 features
        x = jnp.concatenate([x, skip_80], axis=-1)
        
        # Second up-block: 80→160
        x = nn.ConvTranspose(C//4, (4, 4), (2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Third up-block: 160→320
        x = nn.ConvTranspose(C//8, (4, 4), (2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Final 3x3 conv to get heatmap
        x = nn.Conv(1, (3, 3), padding='SAME')(x)
        
        return nn.sigmoid(x)


class SpatiotemporalAttention_temp(nn.Module):
    config: ModelConfig
    
    @nn.compact
    def __call__(self, rgb_frames, mask_frames, *, training=False):
        B, T, H, W, _ = rgb_frames.shape
        
        rgb_encoder = FrameEncoder(name="rgb_enc")
        mask_encoder = FrameEncoder(out_channels=(16, 32, 32), name="mask_enc")
        
        # Define functions with fixed training parameter
        def encode_rgb(frame):
            return rgb_encoder(frame, training=training)
        
        def encode_mask(frame):
            return mask_encoder(frame, training=training)
        
        # ----- Collect encoder feature maps at each level -----
        # We'll track intermediate features from each frame
        rgb_features_all_levels = []
        mask_features_all_levels = []
        
        # Process each frame separately and collect all intermediate features
        for t in range(T):
            # Get features for this frame at all resolution levels
            rgb_frame_features = rgb_encoder.apply_intermediate(
                {'params': self.variables['params']['rgb_enc']}, 
                rgb_frames[:, t], 
                training=training
            )
            
            mask_frame_features = mask_encoder.apply_intermediate(
                {'params': self.variables['params']['mask_enc']}, 
                mask_frames[:, t], 
                training=training
            )
            
            # Append to our per-level lists
            for level_idx, features in enumerate(rgb_frame_features):
                if t == 0:
                    # Initialize the list for this level
                    rgb_features_all_levels.append([])
                    mask_features_all_levels.append([])
                
                rgb_features_all_levels[level_idx].append(features)
                mask_features_all_levels[level_idx].append(mask_frame_features[level_idx])
        
        # Stack frame features at each level into a single tensor (B, T, H', W', C)
        rgb_feature_levels = [jnp.stack(level_features, axis=1) for level_features in rgb_features_all_levels]
        mask_feature_levels = [jnp.stack(level_features, axis=1) for level_features in mask_features_all_levels]
        
        # Concatenate RGB and mask features at each level
        feature_levels = [
            jnp.concatenate([rgb_level, mask_level], axis=-1)
            for rgb_level, mask_level in zip(rgb_feature_levels, mask_feature_levels)
        ]
        
        # Use the final level features for transformer processing
        feats = feature_levels[-1]  # [B, T, H_enc, W_enc, C]
        
        # Reshape to tokens for attention
        B, T, H_enc, W_enc, C = feats.shape
        feats = feats.reshape(B, T * H_enc * W_enc, C)
        
        # Project to embedding dimension
        feats = nn.Dense(self.config.embedding_dim)(feats)
        feats = nn.LayerNorm()(feats)
        
        # Add positional encoding
        pos = self.param("pos_embedding",
                        nn.initializers.normal(0.02),
                        (1, T * H_enc * W_enc, self.config.embedding_dim))
        feats = feats + pos
        
        # Self-attention over the tokens
        attn_output = nn.SelfAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.embedding_dim,
            dropout_rate=self.config.dropout_rate
        )(feats, deterministic=not training)
        
        feats = nn.LayerNorm()(feats + attn_output)
        
        # Project features before decoding
        feats = nn.Dense(self.config.embedding_dim)(feats)
        feats = nn.relu(feats)
        feats = nn.Dropout(self.config.dropout_rate)(feats, deterministic=not training)
        
        # Reshape back to spatial representation
        feats = feats.reshape(B, T, H_enc, W_enc, self.config.embedding_dim)
        
        # Pool over time dimension - mean pooling
        feats = feats.mean(axis=1)  # [B, H', W', C]
        
        # Time-average features at each level for skip connections
        skip_features = [level.mean(axis=1) for level in feature_levels[:-1]]
        
        # U-Net decoder with skip connections
        # Start from bottleneck (40x40)
        x = feats  # [B, 40, 40, embedding_dim]
        
        # First up-block: 40→80 and concatenate with 80x80 features
        x = nn.ConvTranspose(self.config.embedding_dim//2, (4, 4), (2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, skip_features[1]], axis=-1)  # Skip connection
        
        # Second up-block: 80→160 and concatenate with 160x160 features
        x = nn.ConvTranspose(self.config.embedding_dim//4, (4, 4), (2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, skip_features[0]], axis=-1)  # Skip connection
        
        # Final up-block: 160→320
        x = nn.ConvTranspose(self.config.embedding_dim//8, (4, 4), (2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Final 1x1 conv to get heatmap
        x = nn.Conv(1, (3, 3), padding='SAME')(x)
        
        return nn.sigmoid(x)


class SpatiotemporalAttention_old(nn.Module):
    config: ModelConfig
    
    @nn.compact
    def __call__(self, rgb_frames, mask_frames, *, training=False):  # `__call__` not `**call**`
        B, T, H, W, _ = rgb_frames.shape  # Use underscore instead of *
        
        rgb_encoder = FrameEncoder(name="rgb_enc")
        mask_encoder = FrameEncoder(out_channels=(16, 32, 32), name="mask_enc")
        
        # Define functions with fixed training parameter
        def encode_rgb(frame):
            return rgb_encoder(frame, training=training)
        
        def encode_mask(frame):
            return mask_encoder(frame, training=training)
        
        # Apply encoders to each frame
        rgb_feats = jax.vmap(encode_rgb, in_axes=1, out_axes=1)(rgb_frames)  # [B, T, H', W', C1]
        mask_feats = jax.vmap(encode_mask, in_axes=1, out_axes=1)(mask_frames)  # [B, T, H', W', C2]
        
        # Concatenate features along the channel dimension
        feats = jnp.concatenate([rgb_feats, mask_feats], axis=-1)  # [B, T, H', W', C]
        
        # Get dimensions after convolutions
        B, T, H_enc, W_enc, C = feats.shape
        
        # Reshape to tokens: flatten spatial dimensions to tokens
        feats = feats.reshape(B, T * H_enc * W_enc, C)  # [B, T*H'*W', C]
        
        # Project to embedding dimension
        feats = nn.Dense(self.config.embedding_dim)(feats)
        feats = nn.LayerNorm()(feats)
        
        # Add positional encoding - modified for token approach
        pos = self.param("pos_embedding",
                        nn.initializers.normal(0.02),
                        (1, T * H_enc * W_enc, self.config.embedding_dim))
        feats = feats + pos
        
        # Self-attention over the tokens
        attn_output = nn.SelfAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.embedding_dim,
            dropout_rate=self.config.dropout_rate
        )(feats, deterministic=not training)
        
        feats = nn.LayerNorm()(feats + attn_output)
        
        # Project features before decoding
        feats = nn.Dense(self.config.embedding_dim)(feats)
        feats = nn.relu(feats)
        feats = nn.Dropout(self.config.dropout_rate)(feats, deterministic=not training)
        
        # Reshape back to spatial representation
        feats = feats.reshape(B, T, H_enc, W_enc, self.config.embedding_dim)
        
        # Pool over time dimension - mean pooling instead of last frame
        feats = feats.mean(axis=1)  # [B, H', W', C]
        
        # Full 8× decoder
        def up_block(x, out_ch): 
            x = nn.ConvTranspose(out_ch, (4, 4), (2, 2), padding='SAME')(x) 
            x = nn.LayerNorm()(x) 
            return nn.relu(x) 
        
        # Upsampling decoder
        x = up_block(feats, self.config.embedding_dim//2)  # 40→80 
        x = up_block(x, self.config.embedding_dim//4)      # 80→160 
        x = up_block(x, self.config.embedding_dim//8)      # 160→320 
        x = nn.Conv(1, (3, 3), padding='SAME')(x)          # (B, 320, 320, 1) 
        
        return nn.sigmoid(x)  # Return the final heatmap



class Metrics(NamedTuple):
    """Training metrics."""
    loss: float
    rmse: float


def create_train_state(
    config: ModelConfig,
    rng_key: jnp.ndarray,
    learning_rate: float = 1e-4,
    input_shape: Tuple[int, ...] = (1, 5, 320, 320, 3)
) -> train_state.TrainState:
    """Create initial training state with model and optimizer."""
    model = SpatiotemporalAttention(config=config)
    
    # Create dummy inputs
    B, T, H, W, C = input_shape
    dummy_rgb = jnp.ones(input_shape)
    dummy_mask = jnp.ones((B, T, H, W, 1))
    
    # Initialize parameters without batch stats
    variables = model.init(rng_key, dummy_rgb, dummy_mask, training=False)
    
    # Create learning rate schedule with warmup and decay
    warmup_steps = 100
    decay_rate = 0.96
    decay_steps = 500
    
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=learning_rate * 0.1
    )
    
    # Create optimizer with weight decay and gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=schedule_fn)
    )
    
    # Create standard train state without batch stats
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )



def weighted_bce(pred, target, pos_w: float = 10.0, eps: float = 1e-6):
    """
    pred   : sigmoid probabilities  ∈ (0,1)   [B,H,W,1]
    target : {0,1} heat‑map (we treat >0 as 1) [B,H,W,1]
    pos_w  : how many times to up‑weight positives
    """
    target_bin = jnp.where(target > 0.0, 1.0, 0.0)
    pos_mask   = target_bin
    neg_mask   = 1.0 - target_bin

    bce_pos = -jnp.log(pred      + eps) * pos_mask   #   y · log(p)
    bce_neg = -jnp.log(1.0-pred + eps) * neg_mask   # (1‑y)·log(1‑p)

    # weight positives
    loss = pos_w * bce_pos + bce_neg
    return jnp.mean(loss)


@jax.jit
def train_step(
    state: train_state.TrainState,
    rgb_batch: jnp.ndarray,
    mask_batch: jnp.ndarray,
    target_batch: jnp.ndarray,
    rng: jnp.ndarray
) -> Tuple[train_state.TrainState, Metrics, jnp.ndarray]:
    """Perform a single training step with spatial heatmap prediction."""
    # Split random key for dropout
    new_rng, dropout_rng = random.split(rng)
    
    # Define loss function
    def loss_fn(params):
        predictions = state.apply_fn(
            {'params': params},
            rgb_batch, mask_batch,
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        # --- numerically stable BCE with class weighting ---
        epsilon = 1e-7
        pos_weight = 10.0
        
        pos_mask = (target_batch > 0)
        neg_mask = ~pos_mask
        
        # clip predictions to avoid log(0) and log(1)
        pred = jnp.clip(predictions, epsilon, 1.0 - epsilon)
        
        # sums instead of mean, then normalize by count (≥1)
        pos_loss = -pos_weight * jnp.sum(pos_mask * jnp.log(pred)) / jnp.maximum(pos_mask.sum(), 1)
        neg_loss = -jnp.sum(neg_mask * jnp.log1p(-pred)) / jnp.maximum(neg_mask.sum(), 1)
        
        loss = pos_loss + neg_loss
        
        # Optional debugging to detect NaNs early
        # Uncomment this to stop immediately when NaNs appear
        # jax.debug.callback(lambda l: jnp.any(jnp.isnan(l)), loss)
        
        return loss, predictions
    
    # Compute loss and gradients
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Apply gradients
    state = state.apply_gradients(grads=grads)
    
    # Calculate RMSE metric (this doesn't affect training)
    rmse = jnp.sqrt(jnp.mean((predictions - target_batch) ** 2))
    metrics = Metrics(loss=loss, rmse=rmse)
    
    return state, metrics, new_rng


@jax.jit
def train_step_old(
    state: train_state.TrainState,
    rgb_batch: jnp.ndarray,
    mask_batch: jnp.ndarray,
    target_batch: jnp.ndarray,
    rng: jnp.ndarray
) -> Tuple[train_state.TrainState, Metrics, jnp.ndarray]:
    """Perform a single training step with spatial heatmap prediction."""
    # Split random key for dropout
    new_rng, dropout_rng = random.split(rng)
    
    # Define loss function
    def loss_fn(params):
        predictions = state.apply_fn(
            {'params': params},
            rgb_batch, mask_batch,
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        # --- 3) Weighted loss to handle empty frames ---
        # Check if target has any positive pixels
        has_target = jnp.sum(target_batch, axis=(1, 2, 3)) > 0
        
        # Weighted BCE loss - we calculate separately for positive and negative pixels
        epsilon = 1e-7
        pos_weight = 10.0  # Weight for positive pixels
        
        # Positive pixels loss (weighted higher)
        pos_pixels = target_batch > 0
        pos_loss = -pos_weight * jnp.mean(
            target_batch * jnp.log(predictions + epsilon),
            where=pos_pixels
        )
        
        # Negative pixels loss
        neg_pixels = ~pos_pixels
        neg_loss = -jnp.mean(
            (1 - target_batch) * jnp.log(1 - predictions + epsilon),
            where=neg_pixels
        )
        
        # Combined loss
        loss = pos_loss + neg_loss
        
        return loss, predictions
    
    # Rest of function remains the same
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    rmse = jnp.sqrt(jnp.mean((predictions - target_batch) ** 2))
    metrics = Metrics(loss=loss, rmse=rmse)
    
    return state, metrics, new_rng


@jax.jit
def eval_step(
    state: train_state.TrainState,
    rgb_batch: jnp.ndarray,
    mask_batch: jnp.ndarray,
    target_batch: jnp.ndarray
) -> Metrics:
    """
    Perform an evaluation step.
    
    Args:
        state: Current training state
        rgb_batch: Batch of RGB frames [B,T,H,W,3]
        mask_batch: Batch of mask frames [B,T,H,W,1]
        target_batch: Batch of target positions [B,2]
        
    Returns:
        Evaluation metrics
    """
    # Make predictions without batch stats
    predictions = state.apply_fn(
        {'params': state.params},
        rgb_batch, mask_batch,
        training=False
    )
    
    # Calculate metrics
    loss = jnp.mean(jnp.sum((predictions - target_batch) ** 2, axis=-1))
    rmse = jnp.sqrt(jnp.mean(jnp.sum((predictions - target_batch) ** 2, axis=-1)))
    
    return Metrics(loss=loss, rmse=rmse)


def train_model(
    train_dataset_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    val_dataset_fn: Optional[Callable[[], Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = None,
    config: ModelConfig = ModelConfig(),
    num_epochs: int = 3,
    steps_per_epoch: int = 100,
    eval_steps: int = 20,
    learning_rate: float = 1e-4,
    log_every: int = 10,
    save_checkpoint_dir: Optional[str] = None,
    debug_image_dir: Optional[str] = "./out_images",  # Add this parameter
    tensorboard_dir: Optional[str] = "./log_dir",
    resume_checkpoint: Optional[str] = None  # Add this parameter
) -> Dict[str, Any]:
    """
    Train the model using provided dataset functions.
    
    Args:
        train_dataset_fn: Function that returns an iterator of training batches
        val_dataset_fn: Optional function that returns an iterator of validation batches
        config: Model configuration
        num_epochs: Number of training epochs
        steps_per_epoch: Number of steps per epoch
        eval_steps: Number of evaluation steps
        learning_rate: Learning rate
        log_every: Log metrics every N steps
        save_checkpoint_dir: Directory to save checkpoints
        debug_image_dir: Directory to save debug images
        
    Returns:
        Dictionary with trained state and training history
    """
    # Initialize random key
    rng = random.PRNGKey(42)
    rng, init_rng = random.split(rng)

    #Initialize TensorBoard if directory is provided
    summary_writer = None
    if tensorboard_dir:
        time_str = time.strftime("%m%d_%H%M")
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            
            # Create directory if needed
            os.makedirs(tensorboard_dir, exist_ok=True)
            
            # Create SummaryWriter
            summary_writer = SummaryWriter(log_dir=tensorboard_dir + '/' + time_str)
            logger.info(f"TensorBoard logging enabled at {tensorboard_dir}")
        except ImportError:
            logger.warning("Could not import SummaryWriter, TensorBoard logging disabled")

    
    # Create training state (with or without checkpoint)
    logger.info("Initializing model parameters...")
    start_epoch = 0
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        # Load checkpoint
        logger.info(f"Loading checkpoint from {resume_checkpoint}")
        with open(resume_checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Get start epoch
        start_epoch = checkpoint.get('epoch', 0)
        
        # Create optimizer with same configuration
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=learning_rate)
        )
        
        # Create model with same architecture
        model = SpatiotemporalAttention(config=config)
        
        # Create state with existing params and optimizer
        state = train_state.TrainState(
            step=0,  # Will be incremented during training
            apply_fn=model.apply,
            params=checkpoint['params'],
            tx=tx,
            opt_state=checkpoint.get('optimizer_state', None)
        )
        
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        # Create new state
        state = create_train_state(config, init_rng, learning_rate)
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],
        'val_rmse': []
    }
    
    # Training loop - start from the checkpoint epoch
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        train_losses = []
        train_rmses = []
        
        # Create new training dataset iterator
        train_dataset = train_dataset_fn()
        
        # Write debug images for the first batch of each epoch
        try:
            # Get first batch
            first_batch = next(train_dataset)
            rgb_batch, mask_batch, target_batch = first_batch
            
            # Get first example from batch
            rgb_frames = rgb_batch[0]  # [T, H, W, 3]
            mask_frames = mask_batch[0]  # [T, H, W, 1]
            target_heatmap = target_batch[0]  # [H, W, 1]
            
            # Make a prediction using the current model
            rgb_jax = jnp.array(rgb_frames[np.newaxis])  # Add batch dimension
            mask_jax = jnp.array(mask_frames[np.newaxis])  # Add batch dimension
            prediction = state.apply_fn({'params': state.params}, rgb_jax, mask_jax, training=False)
            prediction_np = np.array(prediction[0])  # Remove batch dimension
            
            # For visualization, we'll need to load a future frame
            # Since our NPZ file doesn't store the future frame directly, we'll need to use
            # the target heatmap for visualization or reload from the original dataset
    
            # Option 1: Only use what we already have
            if debug_image_dir and (epoch +1) % 10 == 0:
                # In train_model function, modify the debug visualization call:
                write_debug_images(
                    rgb_frames, 
                    mask_frames,
                    prediction_np,
                    epoch + 1,
                    output_dir=debug_image_dir,
                    target_heatmap=target_heatmap,
                    #dataset_path=dataset_path,  # You need to pass this from train_trajectory_model_efficient
                    #frame_index=0,  # Index of the first frame in the batch
                    #yolo_model_path=yolo_model_path  # Also need to pass this from train_trajectory_model_efficient
                )
            
            # Recreate the training dataset since we consumed one batch
            train_dataset = train_dataset_fn()
        except StopIteration:
            logger.warning("Training dataset empty, cannot write debug images")
            train_dataset = train_dataset_fn()
        
        # Continue with normal training
        for step in range(steps_per_epoch):
            try:
                # Get next batch
                rgb_batch, mask_batch, target_batch = next(train_dataset)
                
                # Convert to JAX arrays
                rgb_jax, mask_jax, target_jax = np_to_jax_batch(rgb_batch, mask_batch, target_batch)
                
                # Perform training step
                rng, step_rng = random.split(rng)
                state, metrics, rng = train_step(state, rgb_jax, mask_jax, target_jax, step_rng)
                # Add debug code here
                if step % 50 == 0:
                    # Get predictions
                    predictions = state.apply_fn(
                        {'params': state.params},
                        rgb_jax, mask_jax,
                        training=False
                    )
                    # Calculate variance along x and y axes
                    """
                    pred = np.array(predictions[0, ..., 0])  # [H,W]
                    print(f"Epoch {epoch+1}, Step {step}")
                    print("σ_x:", pred.var(axis=1).mean(), "σ_y:", pred.var(axis=0).mean())
                    tgt = np.array(target_jax[0, ..., 0])
                    print("tgt σ_x:", tgt.var(axis=1).mean(), "tgt σ_y:", tgt.var(axis=0).mean())
                    """
                # Record metrics
                train_losses.append(float(metrics.loss))
                train_rmses.append(float(metrics.rmse))                
            except StopIteration:
                logger.warning("Training dataset exhausted before completing epoch")
                break
        
        # Calculate epoch metrics
        epoch_loss = np.mean(train_losses) if train_losses else np.nan
        epoch_rmse = np.mean(train_rmses) if train_rmses else np.nan
        
        # Update history
        history['train_loss'].append(float(epoch_loss))
        history['train_rmse'].append(float(epoch_rmse))
        
        # Validation
        if val_dataset_fn is not None:
            val_dataset = val_dataset_fn()
            val_losses = []
            val_rmses = []
            
            for _ in range(eval_steps):
                try:
                    # Get next batch
                    rgb_batch, mask_batch, target_batch = next(val_dataset)
                    
                    # Convert to JAX arrays
                    rgb_jax, mask_jax, target_jax = np_to_jax_batch(rgb_batch, mask_batch, target_batch)
                    
                    # Perform evaluation step
                    metrics = eval_step(state, rgb_jax, mask_jax, target_jax)
                    
                    # Record metrics
                    val_losses.append(float(metrics.loss))
                    val_rmses.append(float(metrics.rmse))
                    
                except StopIteration:
                    logger.warning("Validation dataset exhausted before completing evaluation")
                    break
            
            # Calculate validation metrics
            val_loss = np.mean(val_losses) if val_losses else np.nan
            val_rmse = np.mean(val_rmses) if val_rmses else np.nan
            
            # Update history
            history['val_loss'].append(float(val_loss))
            history['val_rmse'].append(float(val_rmse))
            # Inside the epoch loop, after calculating epoch metrics:
        if summary_writer:
            # Log metrics to TensorBoard
            summary_writer.add_scalar('Loss/train', float(epoch_loss), epoch)
            summary_writer.add_scalar('RMSE/train', float(epoch_rmse), epoch)
        
            if val_dataset_fn is not None and val_losses:
                summary_writer.add_scalar('Loss/val', float(val_loss), epoch)
                summary_writer.add_scalar('RMSE/val', float(val_rmse), epoch)

        # Log epoch summary
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
            f"Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}"
        )
        
        if val_dataset_fn is not None and val_losses:
            logger.info(f"Validation Loss: {val_loss:.4f}, Validation RMSE: {val_rmse:.4f}")
        
        # Save checkpoint
        if (
                save_checkpoint_dir is not None) and (epoch % 10 == 0):
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch + 1,
                'params': state.params,
                'config': config._asdict(),
                'optimizer_state': state.opt_state
            }
            
            checkpoint_path = os.path.join(save_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Close the TensorBoard writer at the end
    if summary_writer:
        summary_writer.close()
            
    # Return final state and history
    return {
        'state': state,
        'history': history,
        'config': config
    }


# 5. Update the prediction function
def predict(
    state: train_state.TrainState,
    rgb_frames: jnp.ndarray,
    mask_frames: jnp.ndarray
) -> jnp.ndarray:
    """
    Make heatmap predictions using trained model.
    
    Args:
        state: Training state with model parameters
        rgb_frames: RGB frames [B,T,H,W,3]
        mask_frames: Mask frames [B,T,H,W,1]
        
    Returns:
        Predicted heatmaps [B,H,W,1]
    """
    # Use the model from the state directly
    return state.apply_fn({'params': state.params}, rgb_frames, mask_frames, training=False)


def train_trajectory_model_efficient(
    preprocessed_train_path: str,
    preprocessed_val_path: str,
    output_dir: str = "./model_output",
    num_epochs: int = 3,
    steps_per_epoch: Optional[int] = None,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    embedding_dim: int = 256,
    num_heads: int = 8,
    debug_image_dir: Optional[str] = "./out_images",
    tensorboard_dir: Optional[str] = "./log_dir", 
    resume_checkpoint: Optional[str] = None,  # Add this parameter
) -> Dict[str, Any]:
    """
    Train the model efficiently using preprocessed datasets.
    
    Args:
        preprocessed_train_path: Path to preprocessed training data
        preprocessed_val_path: Path to preprocessed validation data
        output_dir: Directory to save model and outputs
        num_epochs: Number of training epochs
        steps_per_epoch: Number of steps per epoch (None = full dataset)
        batch_size: Batch size
        learning_rate: Learning rate
        embedding_dim: Embedding dimension
        num_heads: Number of attention heads
        debug_image_dir: Directory to save debug images
        
    Returns:
        Dictionary with trained state and training history
    """
    from trajectory_model import (
        train_model,
        ModelConfig,
        create_train_state,
        predict,
        SpatiotemporalAttention
    )
    
    logging.info(f"Starting efficient training with {num_epochs} epochs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create batch providers
    train_dataset_info = np.load(preprocessed_train_path, mmap_mode='r')
    val_dataset_info = np.load(preprocessed_val_path, mmap_mode='r')
    
    # Get shape information from the datasets
    rgb_shape = train_dataset_info['rgb'].shape
    sequence_length = rgb_shape[1]
    target_height = rgb_shape[2]
    target_width = rgb_shape[3]
    
    # Determine steps_per_epoch if not provided
    if steps_per_epoch is None:
        steps_per_epoch = (rgb_shape[0] + batch_size - 1) // batch_size
    
    # Create batch providers
    train_provider = create_memmap_batch_provider(
        preprocessed_train_path,
        batch_size=batch_size,
        shuffle=True,
        rng_seed=42
    )
    
    val_provider = create_memmap_batch_provider(
        preprocessed_val_path,
        batch_size=batch_size,
        shuffle=False,
        rng_seed=42
    )
    
    # Configure model
    config = ModelConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout_rate=0.1,
        feature_dim=64,
        max_len=5000,
        sequence_length=sequence_length,
        output_height=target_height,
        output_width=target_width
    )
    
    # Add prefetching to the batch providers (optional)
    def prefetch_provider(provider, prefetch_size=2):
        def prefetched_provider():
            for batch in provider():
                # Convert to JAX arrays and prefetch to device
                rgb, mask, target = batch
                rgb_jax = jax.device_put(jnp.array(rgb))
                mask_jax = jax.device_put(jnp.array(mask))
                target_jax = jax.device_put(jnp.array(target))
                yield rgb_jax, mask_jax, target_jax
        return prefetched_provider
    
    # Use prefetching if available
    try:
        train_dataset_fn = prefetch_provider(train_provider)
        val_dataset_fn = prefetch_provider(val_provider)
    except:
        # Fall back to regular providers if prefetching fails
        train_dataset_fn = train_provider
        val_dataset_fn = val_provider
    
    # Train model
    result = train_model(
        train_dataset_fn=train_dataset_fn,
        val_dataset_fn=val_dataset_fn,
        config=config,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        eval_steps=20,
        learning_rate=learning_rate,
        log_every=10,
        save_checkpoint_dir=output_dir,
        tensorboard_dir=tensorboard_dir, 
        debug_image_dir=debug_image_dir,
        resume_checkpoint=resume_checkpoint
    )
    
    # Save final model
    import pickle
    final_model_path = os.path.join(output_dir, "final_model.pkl")
    with open(final_model_path, 'wb') as f:
        pickle.dump({
            'params': result['state'].params,
            'config': config._asdict()
        }, f)
    
    logging.info(f"Final model saved to {final_model_path}")
    
    return result
