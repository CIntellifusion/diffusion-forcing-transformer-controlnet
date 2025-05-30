from typing import Optional
import torch
from torch import Tensor
from omegaconf import DictConfig
from einops import rearrange
from utils.geometry_utils import CameraPose
from .dfot_video import DFoTVideo
from external import VGGTConnector   
from torch.nn import functional  as F 
from typing import Optional, Any, Dict, Literal, Callable, Tuple
from .history_guidance import HistoryGuidance
from tqdm import tqdm
from einops import rearrange, repeat, reduce
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
        base_model_ckpt = cfg.base_model_ckpt 
        self.init_base_model(base_model_ckpt)
    def build_connector(self):
        """
        Build the connector for the model
        """
        self.connector = VGGTConnector(
            hidden_dim=self.cfg.backbone.connector.hidden_dim,
            num_layers=self.cfg.backbone.connector.num_layers,
        )
        # output = connector(images) 
    def init_base_model(self,ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # 纯粹是 state_dict
        # 加载权重
        import pdb; pdb.set_trace() 
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        print(f"[VGGTControlNetDFoTVideoPose][init_base_model] Loaded base model from: {ckpt_path}")
        if missing_keys:
            print(f"[VGGTControlNetDFoTVideoPose][init_base_model] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[VGGTControlNetDFoTVideoPose][init_base_model] Unexpected keys: {unexpected_keys}")
        
    def vggt_processor(self, images: Tensor): 
        # now only use a resize
        # TODO check the correctness 
        # images: b t c h w 
        b = images.shape[0] 
        images = rearrange(images, 'b f c h w -> (b f) c h w')
        images_resized = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        images_resized = rearrange(images_resized, '(b f) c h w -> b f c h w', b=b)
        return images_resized

    # called by 'on_save_checkpoint' 

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) uncompile the model's state_dict before saving
        self._uncompile_checkpoint(checkpoint)
        # 2. Only save the meaningful keys defined by self._should_include_in_checkpoint
        # by default, only the model's state_dict is saved and metrics & registered buffes (e.g. diffusion schedule) are not discarded
        state_dict = checkpoint["state_dict"]
        # rewrite annotation: 
        # since we could not simply re-implement _should_include_in_checkpoint because this functions is shared with _on_load_chekcpoint 
        # we need to re-rewrite on_save_checkpiont
        # during controlnet trianing, we want to load base model state dict and controlnet(optional.)
        # during controlnet inference: we want to load base model and controlnet 
        # during controlnet training, we want to save only trainable(controlnet and connector)
        should_include = [n for n,p in self.named_parameters() if p.requires_grad]
        for key in list(state_dict.keys()):
            if key not in should_include:
                del state_dict[key]
        
    def training_step(self, batch, batch_idx, namespace="training"):
        """Training step"""
        xs, conditions, masks, gt_videos  = batch
        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        # process vggt features n_context 
        context_images = xs[:, : 1, ...]
        context_images = self._unnormalize_x(context_images) # [b, 1, c, h, w]
        vggt_context_images = self.vggt_processor(context_images) 
        control_input = self.connector(vggt_context_images)['features']
        # torch.save(vggt_context_images, "vggt_context_images.pt")
        # print(f"shape of context images {context_images.shape} context image max min {context_images.max()} {context_images.min()}")
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
        # print(f"loss {loss.item()} xs shape {xs.shape} xs_pred shape {xs_pred.shape}")
        return output_dict

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        This function is called by predict sequence and interpolate sequence. 
        We rewrite this function to support vggt conditioning: 
            - we receive context images to input to vggt connector 
            - feed the vggt features to the diffusion model as control input 
    
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        length: Optional[int]
            Number of frames in sampled sequence
            If None, fall back to length of context, and then fall back to `self.max_tokens`
        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
            Context tokens to condition on. Assumed to be same across batch.
            Tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
            Mask for context
            0 = To be generated, 1 = Ground truth context, 2 = Generated context
            Some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.max_tokens (noncausal), ...)
            Unprocessed external conditions for sampling
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        return_all: bool
            Whether to return all steps of the sampling process
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            All recorded intermediate results during the sampling process
        """
        x_shape = self.x_shape

        if length is None:
            length = self.max_tokens if context is None else context.shape[1]
        if length > self.max_tokens:
            raise ValueError(
                f"length is expected to <={self.max_tokens}, got {length}."
            )

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if context.shape[1] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[1]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(
                    f"context shape not compatible with x_stacked_shape {x_shape}."
                )

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        if conditions is not None:
            if self.use_causal_mask and conditions.shape[1] != length:
                raise ValueError(
                    f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
                )
            elif not self.use_causal_mask and conditions.shape[1] != self.max_tokens:
                raise ValueError(
                    f"for noncausal models, conditions length is expected to be {self.max_tokens}, got {conditions.shape[1]}."
                )

        horizon = length if self.use_causal_mask else self.max_tokens
        padding = horizon - length
        # import pdb; pdb.set_trace()
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)
        # import pdb; pdb.set_trace()
        print("[VGGTControlNetDFoTVideoPose] sample sequence function")
        # process vggt features n_context 
        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros_like(
                (batch_size, horizon), dtype=torch.long, device=self.device
            )
            raise NotImplementedError("context is None is not implemented yet for VGGT ControlNet DFoTVideoPose")
        elif padding > 0:
            # pad context and context mask to reach horizon
            context_pad = torch.zeros(
                (batch_size, padding, *x_shape), device=self.device
            )
            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask_pad = -torch.ones(
                (batch_size, padding), dtype=torch.long, device=self.device
            )
            context = torch.cat([context, context_pad], 1)
            context_mask = torch.cat([context_mask, context_mask_pad], 1)
        ### start modification of vggt controlnet dfot video pose
        context_images = torch.stack([context[i, context_mask[i].bool()] for i in range(batch_size)])
        context_images = self._unnormalize_x(context_images)
        vggt_context_images = self.vggt_processor(context_images) 
        control_input = self.connector(vggt_context_images)['features'] # [b, context_length, c, h, w]
        ### end modification of vggt controlnet dfot video pose 
        if history_guidance is None:
            # by default, use conditional sampling
            history_guidance = HistoryGuidance.conditional(
                timesteps=self.timesteps,
            )

        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            horizon - padding,
            padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
        # fill context tokens' noise levels as -1 in scheduling matrix
        if not self.is_full_sequence:
            scheduling_matrix = torch.where(
                context_mask[None] >= 1, -1, scheduling_matrix
            )

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )
        # import pdb; pdb.set_trace()
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1),
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            with history_guidance(context_mask) as history_guidance_manager:
                nfe = history_guidance_manager.nfe
                pbar.set_postfix(NFE=nfe)
                xs_pred, from_noise_levels, to_noise_levels, conditions_mask = (
                    history_guidance_manager.prepare(
                        xs_pred,
                        from_noise_levels,
                        to_noise_levels,
                        replacement_fn=self.diffusion_model.q_sample,
                        replacement_only=self.is_full_sequence,
                    )
                )

                if reconstruction_guidance > 0:

                    def composed_guidance_fn(
                        xk: torch.Tensor,
                        pred_x0: torch.Tensor,
                        alpha_cumprod: torch.Tensor,
                    ) -> torch.Tensor:
                        loss = (
                            F.mse_loss(pred_x0, context, reduction="none")
                            * alpha_cumprod.sqrt()
                        )
                        _context_mask = rearrange(
                            context_mask.bool(),
                            "b t -> b t" + " 1" * len(x_shape),
                        )
                        # scale inversely proportional to the number of context frames
                        loss = torch.sum(
                            loss
                            * _context_mask
                            / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
                        )
                        likelihood = -reconstruction_guidance * 0.5 * loss
                        return likelihood

                else:
                    composed_guidance_fn = guidance_fn

                # update xs_pred by DDIM or DDPM sampling
                xs_pred = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(
                        (
                            repeat(
                                conditions,
                                "b ... -> (b nfe) ...",
                                nfe=nfe,
                            ).clone()
                            if conditions is not None
                            else None
                        ),
                        from_noise_levels,
                    ),
                    conditions_mask,
                    guidance_fn=composed_guidance_fn,
                    control_input=control_input
                )

                xs_pred = history_guidance_manager.compose(xs_pred)

            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(
                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
            )
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record
