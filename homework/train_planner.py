"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Its clobbering time!")

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data


from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .metrics import PlannerMetric  # Provided

import torch.nn.functional as F

def train(
    exp_dir: str = "mlp_logs",
    model_name: str = "mlp_planner",
    lr: float = 0.001,
    num_epoch: int = 50,
    batch_size: int = 128,
    seed: int = 2024,
    long_weight: int = 1,
    lat_weight: int = 1,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else
                          "cpu")
    print(f"Using device: {device}")

    # deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # tensorboard log directory
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    # model
    model = load_model(model_name, **kwargs)
    model = model.to(device)

    # data loaders
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False, batch_size=batch_size, num_workers=2)

    # prediction helper function
    def forward_model(model, model_name, data_dict, device):
      if model_name in ["mlp_planner", "transformer_planner"]:
          left = data_dict["track_left"].to(device)
          right = data_dict["track_right"].to(device)
          return model(left, right)
      elif model_name == "cnn_planner":
          image = data_dict["image"].to(device)
          return model(image)
      
    # Early stopping criteria
    goal = [0.2,0.6] if model_name != "cnn_planner" else [0.3,0.45]

    # loss functions
    def masked_mse_loss(pred, labels, mask): # function for applying mask
      mask = mask.float().unsqueeze(-1)  # (B, 3, 1)
      loss = torch.abs(pred - labels)  # (B, 3, 2)
      # weight coordinates differently
      loss[..., 0] *= long_weight   # longitudinal
      loss[..., 1] *= lat_weight    # lateral
      loss = loss * mask
      return loss.sum() / mask.sum()

    # Choose optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Set training and validation metrics
    global_step = 0
    train_perf = PlannerMetric()
    val_perf = PlannerMetric()
    for epoch in range(num_epoch):
      model.train()
      train_perf.reset()

      for data_dict in train_data:
        # Get target data
        waypoints = data_dict['waypoints'].to(device)
        mask = data_dict['waypoints_mask'].to(device)

        # Zero out gradient
        optimizer.zero_grad()

        # Select correct input and predict
        preds = forward_model(model, model_name, data_dict, device)
           
        # Calculate loss
        loss = masked_mse_loss(
          pred = preds,
          labels = waypoints,
          mask = mask
        )
        loss.backward()
        optimizer.step()

        # Log predictions
        train_perf.add(
          preds = preds,
          labels = waypoints,
          labels_mask = mask
        )

        global_step += 1

      # epoch metrics
      train_dict = train_perf.compute()
      train_acc = train_dict['l1_error']
      train_long = train_dict['longitudinal_error']
      train_lat = train_dict['lateral_error']

      logger.add_scalar('train_longitudinal_error', train_long, epoch)
      logger.add_scalar('train_lateral_error', train_lat, epoch)
      logger.add_scalar('train_accuracy', train_acc, epoch)

      # validation
      model.eval()
      val_perf.reset()

      with torch.inference_mode():
        for data_dict in val_data:
          # Get target data
          waypoints = data_dict['waypoints'].to(device)
          mask = data_dict['waypoints_mask'].to(device)

          # Select correct input and predict
          preds = forward_model(model, model_name, data_dict, device)

          # Log predictions
          val_perf.add(
            preds = preds,
            labels = waypoints,
            labels_mask = mask
          )

      # epoch metrics
      val_dict = val_perf.compute()
      val_acc = val_dict['l1_error']
      val_long = val_dict['longitudinal_error']
      val_lat = val_dict['lateral_error']

      logger.add_scalar('val_longitudinal_error', val_long, epoch)
      logger.add_scalar('val_lateral_error', val_lat, epoch)
      logger.add_scalar('val_accuracy', val_acc, epoch)

      # print on first, last, every 10th epoch
      if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % (num_epoch//10) == 0:
        print(f"Epoch {epoch+1:2d}/{num_epoch:2d} |>")
        print(f">>>  Train - Acc: {train_acc:.2f} | Long: {train_long:.3f} | Lat: {train_lat:.3f} ||")
        print(f">>>  Val --- Acc: {val_acc:.2f} | Long: {val_long:.3f} | Lat: {val_lat:.3f} ||")
        print(f">>>  Goal ------------ | Long < {goal[0]}: {val_long<goal[0]} | Lat < {goal[1]}: {val_lat<goal[1]} ||")
      
      # Early stopping
      if val_long < goal[0] and val_lat < goal[1]:
        print(f">>> Early stopping: goal reached at epoch {epoch}")
        break

    # save model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_lyrs", type=int,nargs="+",
     default=[64,64,64,64])
    

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
