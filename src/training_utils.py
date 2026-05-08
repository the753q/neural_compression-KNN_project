import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import os

class FlopsLimitCallback(pl.Callback):
    """
    Callback that stops training once a certain number of FLOPs has been reached.
    """
    def __init__(self, target_flops):
        super().__init__()
        self.target_flops = target_flops
        self.total_flops = 0
        self.flops_per_step = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.flops_per_step is None:
            self.flops_per_step = self._estimate_flops(pl_module, batch)
            print(f"Estimated FLOPs per training step: {self.flops_per_step:.2e}")

        self.total_flops += self.flops_per_step
        
        # Log the cumulative flops
        pl_module.log("cumulative_flops", float(self.total_flops), on_step=True, on_epoch=False, prog_bar=True)

        if self.total_flops >= self.target_flops:
            trainer.should_stop = True
            print(f"\n[FLOPs Limit] Reached target {self.target_flops:.2e} FLOPs (Current: {self.total_flops:.2e}). Stopping training.")

    def _estimate_flops(self, model, batch):
        """
        Simple heuristic to estimate FLOPs for a single forward + backward pass.
        """
        device = next(model.parameters()).device
        if isinstance(batch, torch.Tensor):
            x = batch.to(device)
        elif isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            return 0

        try:
            # Try to use torch.profiler if available
            from torch.profiler import profile, ProfilerActivity, record_function
            
            # Warmup
            with torch.no_grad():
                model(x)
            
            # Note: with_flops=True requires specific torch versions and may not work on all ops
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                with record_function("model_step"):
                    # We use training_step if available, else forward
                    # training_step includes the loss calculation which is better
                    loss = model.training_step(x, 0)
                    # Backward pass is usually 2x forward
            
            fwd_flops = sum(event.flops for event in prof.key_averages())
            if fwd_flops == 0:
                raise ValueError("Profiler returned 0 flops")
                
            # Total step (forward + backward) is roughly 3x forward
            return fwd_flops * 3
            
        except Exception as e:
            # Fallback to parameter-based estimation (very rough)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # Assuming 128x128 patches and typical conv density
            return num_params * 500 

def universal_train_model(model_class, datamodule, experiment_name, epochs, learning_rate, target_flops=None):
    """
    A universal training function that supports FLOPs limiting.
    """
    model = model_class(learning_rate=learning_rate)
    checkpoint_filename = f"{experiment_name}-{model.name}-best"

    callbacks = []
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    if target_flops is not None:
        flops_callback = FlopsLimitCallback(target_flops=target_flops)
        callbacks.append(flops_callback)
        # When training for flops, we set max_epochs to a very large number
        epochs = 1000 

    csv_logger = CSVLogger("logs/", name=experiment_name)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=csv_logger,
    )

    print("=" * 30)
    print(f"Started experiment: {experiment_name}")
    if target_flops:
        print(f"Targeting computational budget: {target_flops:.2e} FLOPs")
    print(f"Starting training for {model.name}...")
    
    trainer.fit(model, datamodule)

    best_model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)
    return best_model
