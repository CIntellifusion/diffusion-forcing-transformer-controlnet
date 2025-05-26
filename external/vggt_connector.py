import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from external.vggt import load_and_preprocess_images
from external.vggt import VGGT
import glob
import os


def pose_encoding_to_extri_intri(pose_enc, image_size):
    # Placeholder for pose encoding conversion
    batch_size, num_frames = pose_enc.shape[:2]
    extrinsic = torch.zeros(batch_size, num_frames, 4, 4, device=pose_enc.device)
    intrinsic = torch.zeros(batch_size, num_frames, 3, 3, device=pose_enc.device)
    return extrinsic, intrinsic

class VGGTConnector(nn.Module):
    """
    VGGT connector to Diffusion Forcing Transformer Controlnet
    """
    def __init__(self, hidden_dim=512, num_layers=2,out_channels=3):
        super().__init__()
        self.init_vggt_model()
        self.hidden_dim = hidden_dim
        
        # MLP connector to transform VGGT features
        self.connector = nn.Sequential(
            nn.Linear(2048, hidden_dim * 2),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU()) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim * 2, hidden_dim * hidden_dim)
        )
        
        self.out_channels = out_channels 
        # input : [B, F , 1 , H , W ] - > output: [B, F, out_channels, H, W] 
        self.channel_layer = nn.Conv2d(1, out_channels, kernel_size=1, stride=1, padding=0) 
        
    def init_vggt_model(self):
        self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.vggt_model.eval()
        for param in self.vggt_model.parameters():
            param.requires_grad = False
        
    
    def check_image_input(self, x):
        assert x.ndim >= 4, f"Input shape {x.shape} is not valid, should be at least 4D"
        assert x.shape[-3] == 3, f"Input shape {x.shape} is not valid, should have 3 channels"
        assert x.max() <= 1.0, f"Input should be normalized to [0,1], but got {x.max()}"
        assert x.min() >= 0.0, f"Input should be normalized to [0,1], but got {x.min()}"
    
    def forward_model(self, images):
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt_model.shortcut_forward(images)
        return aggregated_tokens_list
    
    @torch.no_grad()
    def encode_vggt_model_with_connector(self, x):
        x = (x + 1.0) / 2.0
        target_size = 518
        b, f = x.shape[:2]
        
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        x = rearrange(x, "(b f) c h w -> b f c h w", b=b, f=f)
        
        self.check_image_input(x)
        
        aggregated_tokens_list = self.forward_model(x)
        
        processed_features = []
        for tokens in aggregated_tokens_list:
            tokens = rearrange(tokens, "b f p d -> (b f p) d")
            features = self.connector(tokens)
            features = rearrange(features, "(b f p) (h1 h2) -> b f p h1 h2", 
                               b=b, f=f, p=tokens.shape[0]//(b*f), h1=self.hidden_dim, h2=self.hidden_dim)
            features = features.mean(dim=2)
            processed_features.append(features)
        # import pdb; pdb.set_trace() 
        final_features = torch.stack(processed_features).mean(dim=0) # b , f , h w 
        b = final_features.shape[0] 
        final_features = rearrange(final_features, "b f h w -> (b f) 1 h w")  # Add channel dimension 
        final_features = self.channel_layer(final_features) 
        final_features = rearrange(final_features, "(b f) c h w -> b f c h w", b=b)
        return {
            "features": final_features,
        }
    
    def forward(self, x):
        return self.encode_vggt_model_with_connector(x)

def main():
    """
    Main function to test VGGTConnector
    """
    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # Initialize connector
    hidden_dim = 512
    connector = VGGTConnector(hidden_dim=hidden_dim).to(device)
    
    # Load and preprocess images
    image_folder = "examples/mix50/images"
    image_names = glob.glob(os.path.join(image_folder, "*"))[:50]
    
    if not image_names:
        raise ValueError(f"No images found in {image_folder}")
    
    try:
        images = load_and_preprocess_images(image_names).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load and preprocess images: {str(e)}")
    
    # Reshape images to (B, F, C, H, W) with B=1
    images = images.unsqueeze(0)  # Add batch dimension
    if images.ndim == 4:
        images = images.unsqueeze(1)  # Add frame dimension if needed
    elif images.ndim == 5 and images.shape[1] != 50:
        # Ensure we have exactly 50 frames
        images = images[:, :50]
    
    # Convert to [-1, 1] range as expected by connector
    images = images * 2.0 - 1.0
    images = images.to(dtype)
    images = images.to(device)
    print(f"Input images shape: {images.shape}")
    
    try:
        # Forward pass through connector
        with torch.cuda.amp.autocast(dtype=dtype):
            output = connector(images)
        
        # Verify output shapes
        print("\nOutput shapes:")
        print(f"Features: {output['features'].shape}")
        print(f"Extrinsic: {output['extrinsic'].shape}")
        print(f"Intrinsic: {output['intrinsic'].shape}")
        print(f"Pose encoding: {output['pose_enc'].shape}")
        
        # Verify expected feature shape
        expected_shape = (1, 50, hidden_dim, hidden_dim)
        assert output['features'].shape == expected_shape, \
            f"Expected feature shape {expected_shape}, got {output['features'].shape}"
        
        print("\nTest passed successfully!")
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()