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
        stage_depth: int = 3,
        stage_width: list = [128,64,64,64]
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
    class Block(torch.nn.Module):
          def __init__(self,input_c,output_c,k_size,stride,n_conv):
            super().__init__()
            self.relu = torch.nn.ReLU()
            padding = (k_size - 1) // 2
            layers = []
            in_c = input_c
            for i in range(n_conv): # Can adjust while training
              s = stride if i == 0 else 1 # only downsample at first conv
              layers.append(torch.nn.Conv2d(in_c,output_c,k_size,s,padding))
              layers.append(torch.nn.ReLU())
              in_c = output_c
            
            self.block = torch.nn.Sequential(*layers)

            if input_c != output_c or stride != 1:
                self.proj = nn.Conv2d(input_c, output_c, kernel_size=1, stride=stride)
            else:
                self.proj = nn.Identity()
          def forward(self,x):
            # Add non-linearity after linear combo
            return self.relu(self.block(x) + self.proj(x)) 
    def __init__(
        self,
        n_waypoints: int = 3,
        in_channels: int = 3,
        k_size: int = 3,
        init_channels: int = 16,
        n_conv: int = 3, # Number of convolutions per block
        n_stages: int = 3, # Number of stages (Channels increase by 2x at each stage)
        stage_size: int = 1 # Number of convolutions per block
    ):
        super(CNNPlanner, self).__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Add first layer
        network = [
          torch.nn.Conv2d(in_channels,init_channels,k_size,padding=(k_size-1)//2),
          torch.nn.ReLU()
          ]

        # Add blocks
        c1 = init_channels
        for _ in range(n_stages):
          c2 = c1*2
          # first block in stage, increase channels + reduce resolution
          network.append(self.Block(c1, c2, k_size,
           stride = 2, n_conv = n_conv))

          # remaining blocks, consistent channels and resolution
          for _ in range(stage_size - 1):
              network.append(self.Block(c2, c2, k_size, 1, n_conv))
          
          # Next stage input size
          c1 = c2

        # Add 1x1 conv as classifier
        network.append(torch.nn.Conv2d(c1,2 * n_waypoints,1))

        # Add GAP for selection
        network.append(torch.nn.AdaptiveAvgPool2d((1, 1)))

        self.ResNet = torch.nn.Sequential(*network)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        out = self.ResNet(x)
        return out.view(x.size(0), self.n_waypoints, 2)   


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
