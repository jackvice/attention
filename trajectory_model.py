import os
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

# Configure logging
logger = logging.getLogger("trajectory_model")
    
class FrameEncoder(nn.Module):
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

class SpatiotemporalAttention_old(nn.Module):
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
        
        # Pool over time dimension (alternative to mean pooling)
        # Use attention pooling or just take the last time step
        # Simple version: just use the last frame
        #feats = feats[:, -1]  # [B, H', W', C]

        x = nn.ConvTranspose(
            features=self.config.embedding_dim // 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME"
        )(feats)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.ConvTranspose(
            features=self.config.embedding_dim // 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME"
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Final convolution to get single-channel heatmap
        x = nn.Conv(
            features=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        
        # Ensure output is the correct size
        if x.shape[1:3] != (self.config.output_height, self.config.output_width):
            # Resize if needed - this is a fallback solution
            x = jax.image.resize(
                x, 
                shape=(B, self.config.output_height, self.config.output_width, 1),
                method="bilinear"
            )
        
        return nn.sigmoid(x)  # Output shape: [B, H, W, 1]



class SpatiotemporalAttention_old(nn.Module):
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

        # vmap over time axis → shape (B, T, C)
        rgb_feats = jax.vmap(encode_rgb, in_axes=1, out_axes=1)(rgb_frames)
        mask_feats = jax.vmap(encode_mask, in_axes=1, out_axes=1)(mask_frames)

        # concat & project
        feats = jnp.concatenate([rgb_feats, mask_feats], axis=-1)
        feats = nn.Dense(self.config.embedding_dim)(feats)
        feats = nn.LayerNorm()(feats)

        # + positional encoding
        pos = self.param("pos_embedding",
                         nn.initializers.normal(0.02),
                         (1, T, self.config.embedding_dim))
        feats = feats + pos

        # self-attention + residual
        attn = nn.SelfAttention(num_heads=self.config.num_heads,
                                qkv_features=self.config.embedding_dim,
                                dropout_rate=self.config.dropout_rate)(
                   feats, deterministic=not training)
        feats = nn.LayerNorm()(feats + attn)

        pooled = jnp.mean(feats, axis=1)
        x = nn.Dense(self.config.embedding_dim)(pooled)
        x = nn.relu(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.config.embedding_dim // 2)(x)
        x = nn.relu(x)
        # New heatmap output layers
        x = nn.Dense(self.config.output_height * self.config.output_width)(x)  # Flattened spatial output
        x = jnp.reshape(x, (-1, self.config.output_height, self.config.output_width, 1))  # Reshape to spatial dimensions
        return nn.sigmoid(x)  # (B, H, W, 1) heatmap with values in 0…1


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


import jax
import jax.numpy as jnp

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


# 3. Modify the train_step function to handle spatial heatmaps
@jax.jit
def train_step_old(
    state: train_state.TrainState,
    rgb_batch: jnp.ndarray,
    mask_batch: jnp.ndarray,
    target_batch: jnp.ndarray,  # Now [B, H, W, 1]
    rng: jnp.ndarray
) -> Tuple[train_state.TrainState, Metrics, jnp.ndarray]:
    """
    Perform a single training step with spatial heatmap prediction.
    """
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
        
        # MSE loss for spatial heatmap
        mse_loss = jnp.mean((predictions - target_batch) ** 2)
        
        
        # Optional: KL Divergence for probability distribution comparison
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        p_true = target_batch + epsilon
        p_pred = predictions + epsilon
        
        # Normalize to ensure they're probability distributions
        p_true = p_true / jnp.sum(p_true, axis=(1, 2, 3), keepdims=True)
        p_pred = p_pred / jnp.sum(p_pred, axis=(1, 2, 3), keepdims=True)
        
        kl_loss = jnp.mean(jnp.sum(p_true * jnp.log(p_true / p_pred), axis=(1, 2, 3)))
        
        # Combined loss
        loss = mse_loss + 0.1 * kl_loss  # Weight KL loss to balance with MSE #jmv
        #loss = weighted_bce(p_pred, target_batch, pos_w=10.0) #jmv
        return loss, predictions
    
    # Compute gradients
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Compute metrics
    # For heatmaps, we might use different metrics
    # Here we're keeping RMSE but it's computed differently
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
    debug_image_dir: Optional[str] = "./out_images"  # Add this parameter
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
    
    # Create training state
    logger.info("Initializing model parameters...")
    state = create_train_state(config, init_rng, learning_rate)
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],
        'val_rmse': []
    }
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
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
            if debug_image_dir:
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
                    pred = np.array(predictions[0, ..., 0])  # [H,W]
                    print(f"Epoch {epoch+1}, Step {step}")
                    print("σ_x:", pred.var(axis=1).mean(), "σ_y:", pred.var(axis=0).mean())
                    tgt = np.array(target_jax[0, ..., 0])
                    print("tgt σ_x:", tgt.var(axis=1).mean(), "tgt σ_y:", tgt.var(axis=0).mean())
                # Record metrics
                train_losses.append(float(metrics.loss))
                train_rmses.append(float(metrics.rmse))
                
                # Log progress
                if False:#(step + 1) % log_every == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{steps_per_epoch}, "
                        f"Loss: {metrics.loss:.4f}, RMSE: {metrics.rmse:.4f}"
                    )
                
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
            
        # Save visualization of last frame
        if False:#save_checkpoint_dir is not None:
            # Get last frame from validation set if available
            try:
                import os
                if val_dataset_fn is not None:
                    val_dataset = val_dataset_fn()
                    last_rgb_batch, last_mask_batch, _ = next(val_dataset)
                    
                    # Get last image from batch
                    last_image = last_rgb_batch[0, -1]  # Last frame of first sequence
                    
                    # Detect pedestrians
                    from trajectory_utils import detect_pedestrians_yolo_onnx, visualize_and_save_detections
                    pedestrians, _ = detect_pedestrians_yolo_onnx(
                        last_image, 
                        session=ort.InferenceSession(
                            "/home/jack/src/attention/models/yolo11n.onnx",
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                    )
            
                    # Save visualization
                    output_path = os.path.join(save_checkpoint_dir, f"people_epoch_{epoch+1}.png")
                    visualize_and_save_detections(last_image, pedestrians, output_path)
            except Exception as e:
                logger.error(f"Error saving visualization: {e}")

        # Log epoch summary
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
            f"Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}"
        )
        
        if val_dataset_fn is not None and val_losses:
            logger.info(f"Validation Loss: {val_loss:.4f}, Validation RMSE: {val_rmse:.4f}")
        
        # Save checkpoint
        if save_checkpoint_dir is not None:
            import os
            import pickle
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


