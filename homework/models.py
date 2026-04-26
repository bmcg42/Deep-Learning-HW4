from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        stage_depth: int = 1,
        stage_width: list = [64,64,64]
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        class Block(torch.nn.Module): # Define block of layers
          def __init__(self, in_channels,out_channels):
              super().__init__()
              self.conv = torch.nn.Linear(in_channels,out_channels)
              self.norm = torch.nn.LayerNorm(out_channels)
              self.relu = torch.nn.ReLU()
              # Check if skip connection is neccessary
              if in_channels != out_channels:
                  self.skip = torch.nn.Linear(in_channels,out_channels)
              else:
                  self.skip = torch.nn.Identity()

          def forward(self,x):
              y = self.conv(x)
              y = self.norm(y)
              y = self.relu(y)
              return y + self.skip(x)

        super(MLPPlanner,self).__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        

        # Input dimension
        c = 8 * self.n_track 
        self.input_norm = nn.LayerNorm(c) # input normalization

        # Create embedding layer
        embedding = stage_width[0]
        layers_ls = [torch.nn.Linear(c,embedding)]
        c = embedding

        # Add stages at each specified width
        for i, s in enumerate(stage_width):
          if i == 0: # First layer
              layers_ls.append(nn.Linear(c, s))
          else: # Transition stage
              layers_ls.append(nn.Linear(prev_s, s))
          # Add stages at designated depth
          for _ in range(stage_depth):
              layers_ls.append(Block(s, s))

          prev_s = s

        # Add layer to format output to correct size
        layers_ls.append(nn.Linear(prev_s, self.n_waypoints * 2)) # waypoints x 2 coords

        # Compile layers
        self.model = torch.nn.Sequential(*layers_ls)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        x = torch.cat([
          track_left,
          track_right,
          (track_left + track_right) / 2, # midpoint
          track_left - track_right], # width
          dim=1)  # (B, 2 * n_track, 2)
        x = x.flatten(start_dim=1)           # (B, 4 * n_track)
        x = self.input_norm(x) # input normalization
        out = self.model(x)
        return out.view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        raise NotImplementedError


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
