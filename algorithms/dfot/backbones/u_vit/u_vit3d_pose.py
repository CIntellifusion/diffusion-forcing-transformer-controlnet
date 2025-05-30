from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from omegaconf import DictConfig
from einops import rearrange
from ..modules.embeddings import (
    RandomDropoutPatchEmbed,
)
from .u_vit3d import UViT3D


class UViT3DPose(UViT3D):
    """
    U-ViT with pose embedding.
    """

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        self.conditioning_dropout = cfg.external_cond_dropout
        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            cfg.conditioning.dim,
            use_causal_mask,
        )

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        return RandomDropoutPatchEmbed(
            dropout_prob=self.conditioning_dropout,
            img_size=self.x_shape[1],
            patch_size=self.cfg.patch_size,
            in_chans=self.external_cond_dim,
            embed_dim=self.external_cond_emb_dim,
            bias=True,
            flatten=False,
        )

    def _rearrange_and_add_pos_emb_if_transformer(
        self, x: Tensor, emb: Tensor, i_level: int
    ) -> Tuple[Tensor, Tensor]:
        is_transformer = self.is_transformers[i_level]
        if not is_transformer:
            return x, emb
        x, emb = map(
            lambda y: rearrange(
                y, "(b t) c h w -> b (t h w) c", t=self.temporal_length
            ),
            (x, emb),
        )
        if self.pos_emb_type == "learned_1d":
            x = self.pos_embs[f"{i_level}"](x)
        return x, emb

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
        control_input: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of the U-ViT backbone, with pose conditioning.
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            noise_levels: Noise level tensor of shape (B, T).
            external_cond: External conditioning tensor of shape (B, T, C', H, W).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """
        assert control_input is None, "Control input is not supported in U-ViT3DPose model. Just for dummy interface compatibility."
        # import pdb; pdb.set_trace()
        assert (
            x.shape[1] == self.temporal_length
        ), f"Temporal length of U-ViT is set to {self.temporal_length}, but input has temporal length {x.shape[1]}."

        assert (
            external_cond is not None
        ), "External condition (camera pose) is required for U-ViT3DPose model."

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)

        # Embeddings
        external_cond = self.external_cond_embedding(external_cond, external_cond_mask)
        emb = self.noise_level_pos_embedding(noise_levels)
        emb = rearrange(
            rearrange(emb, "b t c -> b t c 1 1") + external_cond,
            "b t c h w -> (b t) c h w",
        )

        # Down-sample embeddings for each level
        embs = [
            (
                emb
                if i_level == 0
                # pylint: disable-next=not-callable
                else F.avg_pool2d(emb, kernel_size=2**i_level, stride=2**i_level)
            )
            for i_level in range(self.num_levels)
        ]
        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling
        # print(f"original shape x: {x.shape}") #  x: torch.Size([64, 128, 128, 128])
        # Down-sampling blocks
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            x = self._run_level(x, embs[i_level], i_level)
            # print(f"down sample i_level: {i_level}, x.shape: {x.shape}")
            hs_before.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        # Middle blocks
        x = self._run_level(x, embs[-1], self.num_levels - 1)
        
        # import pdb; pdb.set_trace()
        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, embs[i_level], i_level, is_up=True)
            # print(f"upsample i_level: {i_level}, x.shape: {x.shape}")
        #down sample i_level: 0, x.shape: torch.Size([64, 128, 128, 128])
        # down sample i_level: 1, x.shape: torch.Size([64, 256, 64, 64])
        # down sample i_level: 2, x.shape: torch.Size([64, 576, 32, 32])
        #upsample i_level: 2, x.shape: torch.Size([64, 576, 32, 32])
        # upsample i_level: 1, x.shape: torch.Size([64, 256, 64, 64])
        # upsample i_level: 0, x.shape: torch.Size([64, 128, 128, 128])
        # import pdb; pdb.set_trace()
        x = self.project_output(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
    
from typing import Optional, Tuple
from functools import partial
from omegaconf import DictConfig
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from ..base_backbone import BaseBackbone
from ..modules.embeddings import RotaryEmbedding3D
from ..dit.dit_base import SinusoidalPositionalEmbedding
from .u_vit_blocks import (
    EmbedInput,
    ProjectOutput,
    ResBlock,
    TransformerBlock,
    Upsample,
    Downsample,
    AxialRotaryEmbedding,
)

import torch
from torch import nn
from copy import deepcopy
from functools import partial


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetUViT3DPose(nn.Module):
    def __init__(self, 
                cfg: DictConfig,
                x_shape: torch.Size,
                max_tokens: int,
                external_cond_dim: int,
                use_causal_mask=True,
                base_model: Optional[UViT3DPose] = None, 
                train_base: bool = False, 
                train_control: bool = True
                ):
        super().__init__()

        # Base model construction
        if base_model is None:
            self.base_model = UViT3DPose(cfg, x_shape, max_tokens, external_cond_dim, use_causal_mask)
        else:
            self.base_model = base_model

        # Match config from base model
        channels = cfg.channels
        block_types = cfg.block_types
        block_dropouts = cfg.block_dropouts
        num_updown_blocks = cfg.num_updown_blocks
        num_levels = len(channels)
        num_heads = cfg.num_heads
        pos_emb_type = cfg.pos_emb_type
        temporal_length = max_tokens
        emb_dim = cfg.emb_channels

        # Helper for RoPE kwargs
        def _rope_kwargs(i_level: int):
            is_transformer = block_types[i_level] != "ResBlock"
            if pos_emb_type != "rope" or not is_transformer:
                return {}
            return {"rope": self.base_model.pos_embs[f"{i_level}"]}

        # Create parallel control blocks
        self.controlnet_use_checkpointing = list(cfg.controlnet_use_checkpointing)
        self.control_down_blocks = nn.ModuleList()
        self.control_projections = nn.ModuleList()
        # TODO modify this later 
        self.before_proj = zero_module(nn.Linear(channels[0],channels[0]))
        block_type_to_cls = {
            "ResBlock": partial(ResBlock, emb_dim=emb_dim),
            "TransformerBlock": partial(TransformerBlock, emb_dim=emb_dim, heads=num_heads),
            "AxialTransformerBlock": partial(
                TransformerBlock,
                emb_dim=emb_dim,
                heads=num_heads,
                use_axial=True,
                ax1_len=temporal_length,
            ),
        }

        for i_level, (num_blocks, ch, block_type, block_dropout) in enumerate(
            zip(num_updown_blocks, channels[:-1], block_types[:-1], block_dropouts[:-1])
        ):
            # Create control block (excluding Downsample)
            block_modules = nn.ModuleList(
                [
                    block_type_to_cls[block_type](
                        ch, dropout=block_dropout, **_rope_kwargs(i_level)
                    )
                    for _ in range(num_blocks)
                ]
                + [Downsample(ch, channels[i_level + 1])]
            )
            self.control_down_blocks.append(block_modules)
            ch = channels[i_level + 1]
            # 1x1 Conv projection, zero-init
            proj = nn.Conv2d(ch, ch, kernel_size=1)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.control_projections.append(proj)
        # import pdb; pdb.set_trace()
        # print(self.state_dict().keys())
        # Set trainability
        self.set_trainable(train_base=train_base, train_control=train_control)
    
    def init_from_ckpt(self, ckpt_path):
        """
        Initialize the ControlNet model from a checkpoint.
        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        if "controlnet" in ckpt_path:
            self.init_from_controlnet(ckpt_path)
        else:
            self.init_from_base_model(ckpt_path)
        
        return 

    def init_from_base_model(self, ckpt_path):
        """
        Initialize the ControlNet model from a checkpoint of the base model.
        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        # Load the state dict from the checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # Load weights into base_model and print missing/unexpected keys
        result = self.base_model.load_state_dict(state_dict, strict=False,weights_only=False)

        if result.missing_keys:
            print("Missing keys (not found in checkpoint):")
            for k in result.missing_keys:
                print(f"  - {k}")

        if result.unexpected_keys:
            print("Unexpected keys (not used in base_model):")
            for k in result.unexpected_keys:
                print(f"  - {k}")
    
    def init_from_controlnet(self, ckpt_path):
        """
        Initialize the ControlNet model from a checkpoint of the ControlNet model.
        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        # Load the state dict from the checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # Load the modified state dict into the ControlNet model
        result = self.load_state_dict(state_dict, strict=False,weights_only=False)

        if result.missing_keys:
            print("Missing keys (not found in checkpoint):")
            for k in result.missing_keys:
                if "base_model" not in k:
                    print(f"  - {k}")

        if result.unexpected_keys:
            print("Unexpected keys (not used in base_model):")
            for k in result.unexpected_keys:
                if "base_model" not in k:
                    print(f"  - {k}")
                
    def self_test(self):
        # the output of this model should be the same as the original model at the first level
        x = torch.randn(2, 8, 3, 256, 256).to(self.base_model.device)
        noise_levels = torch.randn(2, 8).to(self.base_model.device)
        control_input = None 
        external_cond = torch.randn(2, 8, 3, 256, 256).to(self.base_model.device)
        external_cond_mask = torch.randint(0,2,[2]).to(self.base_model.device)  
        base_output = self.base_model(x, noise_levels, external_cond, external_cond_mask)
        control_output = self.forward(x, noise_levels, control_input, external_cond, external_cond_mask)
        assert base_output.shape == control_output.shape, f"Output shape mismatch: {base_output.shape} vs {control_output.shape}"
        assert torch.allclose(base_output, control_output, atol=1e-5), "Output mismatch between base and control models"
        print("ControlNetUViT3D self-test passed.")
        
    def set_trainable(self, train_base: bool = False, train_control: bool = True):
        """
        Sets which parts of the network should be trainable.

        Args:
            train_base (bool): Whether the base UViT3D model should be trainable.
            train_control (bool): Whether the ControlNet control layers should be trainable.
        """

        # Base UViT3D
        for param in self.base_model.parameters():
            param.requires_grad = train_base

        # Control blocks
        for control_block in self.control_down_blocks:
            for block in control_block:
                for param in block.parameters():
                    param.requires_grad = train_control

        # 1x1 projection layers
        for proj in self.control_projections:
            for param in proj.parameters():
                param.requires_grad = train_control
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) 
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[ControlNetUViT3DPose] Trainable parameters: {trainable_params}, Frozen parameters: {frozen_params}")
    def run_controlnet_level_blocks(
        self, x: Tensor, emb: Tensor, i_level: int, is_up: bool = False
    ) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level.
        Gradient checkpointing is used optionally, with self.checkpoints[i_level] segments.
        """
        use_checkpointing = self.controlnet_use_checkpointing[i_level]
        # shares the same index with down_blocks in _run_level_blocks
        blocks = self.control_down_blocks[i_level][:-1]
        for block in blocks:
            x = self.base_model._checkpointed_forward(
                block,
                x,
                emb,
                use_checkpointing=use_checkpointing,
            )
        return x

    def _run_controlnet_level(
        self, x: Tensor, emb: Tensor, i_level: int, is_up: bool = False
    ) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level, accompanied by reshaping operations before and after.
        """
        x, emb = self.base_model._rearrange_and_add_pos_emb_if_transformer(x, emb, i_level)
        x = self.base_model._run_level_blocks(x, emb, i_level, is_up)
        x = self.base_model._unrearrange_if_transformer(x, i_level)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        control_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [b,3,256,256]
        assert (
            x.shape[1] == self.base_model.temporal_length
        ), f"Temporal length of U-ViT is set to {self.base_model.temporal_length}, but input has temporal length {x.shape[1]}."

        assert (
            external_cond is not None
        ), "External condition (camera pose) is required for U-ViT3DPose model."

        # import pdb; pdb.set_trace()
        # Initial reshaping and embedding
        x = rearrange(x, "b t c h w -> (b t) c h w")
        if control_input is None:
            print("[Warning][ControlNetUViT3DPose]: control_input is None, using x as control input")
            control_input = x.clone() 
        elif control_input.shape[1]  != self.base_model.temporal_length: 
            # repeat
            # import pdb; pdb.set_trace() 
            control_input_length = control_input.shape[1]
            if control_input_length != 1:
                # print(f"[Warning][ControlNetUViT3DPose] Control input length is {control_input_length}, means multiple vggt referecing frames, reducing to 1 by mean pooling")
                # control_input = torch.mean(control_input, dim=1, keepdim=True)  # mean pooling
                # print(f"[Warning][ControlNetUViT3DPose] Control input length is {control_input_length}, means multiple vggt referecing frames, reducing to 1 by taking the first frame")
                control_input = control_input[:,:1,...] # first frame 
                control_input_length = control_input.shape[1]
            assert control_input_length == 1, f"[Error][ControlNetUViT3DPose]Control input length should be 1, but got {control_input_length}"
            control_input = control_input.repeat(1, self.base_model.temporal_length, 1, 1, 1)
            # Combine batch and temporal dimensions
        control_input = rearrange(control_input, "b t c h w -> (b t) c h w")
        control_input = self.base_model.embed_input(control_input) 
        x = self.base_model.embed_input(x)
        control_input = self.before_proj(control_input) # before proj should have same size with embed input seq length

        # Embeddings
        external_cond = self.base_model.external_cond_embedding(external_cond, external_cond_mask)
        emb = self.base_model.noise_level_pos_embedding(noise_levels)
        emb = rearrange(
            rearrange(emb, "b t c -> b t c 1 1") + external_cond,
            "b t c h w -> (b t) c h w",
        )
        # Down-sample embeddings for each level
        embs = [
            (
                emb
                if i_level == 0
                # pylint: disable-next=not-callable
                else F.avg_pool2d(emb, kernel_size=2**i_level, stride=2**i_level)
            )
            for i_level in range(self.base_model.num_levels)
        ]
        # import pdb; pdb.set_trace()
        # print([emb.shape for emb in embs])
        hs_before = []
        hs_after = []
        control_feats_before = []
        control_feats_after = []
        # import pdb; pdb.set_trace()
        control_input = x + control_input 
        # Down blocks
        for i_level, (down_block, ctrl_block, proj) in enumerate(
            zip(self.base_model.down_blocks,
                self.control_down_blocks,
                self.control_projections)
        ):
            # Base path
            # import pdb; pdb.set_trace()
            # print(f"ievel {i_level} {embs[i_level].shape}")
            x = self.base_model._run_level(x,embs[i_level], i_level)
            hs_before.append(x)
            x = down_block[-1](x)  # Downsample
            hs_after.append(x)
            # Control path (no downsampling)
            # CHECK here agiain 
            control_input = self._run_controlnet_level(control_input, embs[i_level], i_level)
            control_input = ctrl_block[-1](control_input)  # Downsample
            control_feat = proj(control_input)
            control_feats_after.append(control_feat)
            # import pdb; pdb.set_trace()
            # print(f"last in hs_before shape {hs_before[-1].shape}, hs_after shape {hs_after[-1].shape}, control_feats shape {control_feats_after[-1].shape}")
        # import pdb; pdb.set_trace()
        # Mid
        x = self.base_model._run_level(x, embs[-1], self.base_model.num_levels - 1)
        assert len(hs_before) == len(hs_after)  == len(control_feats_after) , f"hs_before: {len(hs_before)}, hs_after: {len(hs_after)}, control_feats: {len(control_feats_after)}"
        # Up blocks
        for _i_level, up_block in enumerate(self.base_model.up_blocks):
            i_level = self.base_model.num_levels - 2 - _i_level
            x = x - (hs_after.pop()+control_feats_after.pop())
            x = up_block[0](x) + hs_before.pop()
            x = self.base_model._run_level(x, embs[i_level] , i_level, is_up=True)

        # Final projection
        x = self.base_model.project_output(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=self.base_model.temporal_length)
