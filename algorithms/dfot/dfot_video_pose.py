from typing import Optional
import torch
from torch import Tensor
from omegaconf import DictConfig
from einops import rearrange
from utils.geometry_utils import CameraPose
from .dfot_video import DFoTVideo
from external import VGGTConnector   

class DFoTVideoPose(DFoTVideo):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for pose-conditioned video generation.
    """

    def __init__(self, cfg: DictConfig):
        self.camera_pose_conditioning = cfg.camera_pose_conditioning
        self.conditioning_type = cfg.camera_pose_conditioning.type
        self._check_cfg(cfg)
        self._update_backbone_cfg(cfg)
        super().__init__(cfg)
        # import pdb; pdb.set_trace()
        
    def _check_cfg(self, cfg: DictConfig):
        """
        Check if the config is valid
        """
        if cfg.backbone.name not in {"dit3d_pose", "u_vit3d_pose","u_vit3d_pose_controlnet"}:
            raise ValueError(
                f"DiffusionForcingVideo3D only supports backbone 'dit3d_pose' or 'u_vit3d_pose_controlnet' 'dit3d_pose_controlnet', got {cfg.backbone.name}"
            )

        if (
            cfg.backbone.name == "dit3d_pose"
            and self.conditioning_type == "global"
            and cfg.backbone.conditioning.modeling != "film"
        ):
            raise ValueError(
                f"When using global camera pose conditioning, `algorithm.backbone.conditioning.modeling` should be 'film', got {cfg.backbone.conditioning.modeling}"
            )
        if (cfg.backbone.name in ["u_vit3d_pose","u_vit3d_pose_controlnet"]) and self.conditioning_type == "global":
            raise ValueError(
                "Global camera pose conditioning is not supported for U-ViT3DPose"
            )
        ## adding additional control to diffusion backbone 

    def _update_backbone_cfg(self, cfg: DictConfig):
        """
        Update backbone config with camera pose conditioning
        """
        conditioning_dim = None
        match self.conditioning_type:
            case "global":
                conditioning_dim = 12
            case "ray" | "plucker":
                conditioning_dim = 6
            case "ray_encoding":
                conditioning_dim = 180
            case _:
                raise ValueError(
                    f"Unknown camera pose conditioning type: {self.conditioning_type}"
                )
        cfg.backbone.conditioning.dim = conditioning_dim

    @torch.no_grad()
    @torch.autocast(
        device_type="cuda", enabled=False
    )  # force 32-bit precision for camera pose processing
    def _process_conditions(
        self, conditions: Tensor, noise_levels: Optional[Tensor] = None
    ) -> Tensor:
        """
        Process conditions (raw camera poses) to desired format for the model
        Args:
            conditions (Tensor): raw camera poses (B, T, 12)
        """
        camera_poses = CameraPose.from_vectors(conditions)
        if self.cfg.tasks.prediction.history_guidance.name == "temporal":
            # NOTE: when using temporal history guidance,
            # some frames are fully masked out and thus their camera poses are not needed
            # so we replace them with interpolated camera poses from the nearest non-masked frames
            # this is important b/c we normalize camera poses by the first frame
            camera_poses.replace_with_interpolation(
                mask=noise_levels == self.timesteps - 1
            )

        match self.camera_pose_conditioning.normalize_by:
            case "first":
                camera_poses.normalize_by_first()
            case "mean":
                camera_poses.normalize_by_mean()
            case _:
                raise ValueError(
                    f"Unknown camera pose normalization method: {self.camera_pose_conditioning.normalize_by}"
                )

        if self.camera_pose_conditioning.bound is not None:
            camera_poses.scale_within_bounds(self.camera_pose_conditioning.bound)

        match self.conditioning_type:
            case "global":
                return camera_poses.extrinsics(flatten=True)
            case "ray" | "ray_encoding" | "plucker":
                rays = camera_poses.rays(resolution=self.x_shape[1])
                if self.conditioning_type == "ray_encoding":
                    rays = rays.to_pos_encoding()[0]
                else:
                    rays = rays.to_tensor(
                        use_plucker=self.conditioning_type == "plucker"
                    )
                return rearrange(rays, "b t h w c -> b t c h w")

      
class VGGTControlNetDFoTVideoPose(DFoTVideoPose):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for pose-conditioned video generation.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.build_connector()

    def build_connector(self):
        """
        Build the connector for the model
        """
        self.connector = VGGTConnector(
            hidden_dim=self.cfg.backbone.connector.hidden_dim,
            num_layers=self.cfg.backbone.connector.num_layers,
        )
        # output = connector(images) 
    def vggt_processor(self, images: Tensor): 
        # now only use a resize
        # TODO check the correctness 
        # images: b t c h w 
        from torch.nn import functional  as F 
        b = images.shape[0] 
        images = rearrange(images, 'b f c h w -> (b f) c h w')
        images_resized = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        images_resized = rearrange(images_resized, '(b f) c h w -> b f c h w', b=b)
        return images_resized
    
    def training_step(self, batch, batch_idx, namespace="training"):
        """Training step"""
        xs, conditions, masks, gt_videos  = batch

        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        # process vggt features n_context 
        context_images = gt_videos[:, : 1, ...]
        vggt_context_images = self.vggt_processor(context_images) 
        control_input = self.connector(vggt_context_images)['features']
        # torch.save(vggt_context_images, "vggt_context_images.pt")
        print(f"shape of context images {context_images.shape} context image max min {context_images.max()} {context_images.min()}")
        # import pdb; pdb.set_trace()
        # to continuouse diffusion 
        xs_pred, loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
            control_input=control_input
        )
        loss = self._reweight_loss(loss, masks)

        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict
