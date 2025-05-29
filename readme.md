# Dfot-VGGT

the original readme: dfot-readme.md
dataset process: prepare-re10k-mp4.md

## Prepare DFOT-RE10k dataset for training

## Why Give Up from ControlNet on Hunyuan Video

Poor Spatial Understanding: Hunyuan Video lacks the spatial reasoning capabilities required for real-world applications. In contrast, Dfot is better aligned with practical deployment scenarios.

Model Size Constraints: Hunyuan Video is already a large model, and integrating an additional 1.5B-parameter online VGGT model would exceed feasible capacity limits.

Planning Over Quality: We use Dfot v1 for planning and robotics tasks, rather than v2, as our focus is not on achieving higher video quality. Dfot v1 serves as a sufficient baseline and naturally supports autoregressive video generation.

## Plan

- [ ] run inference and evaluate of long video generation
- [ ] add a controlnet finetuning task
  - [ ] finetune on a pretrained model
  - [ ] eval long video generation with controlnet
- [ ] use a learnable token to inject vggt

## experiment setup

We aims to improve video consistency by vggt latent features.

We benchmark basic dfot for long video generation and expect improvement after vggt post-training.

## contorlnet

design checklist:

- [x] same initialization parametere
- [x] same output from each downsampling blocks
- [x] zero-init
- [x] forward

##

Re10k has a UViT3DPose as backbone. in forward blocks,

```python
        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling

        # Down-sampling blocks
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            x = self._run_level(x, embs[i_level], i_level)
            hs_before.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        # Middle blocks
        x = self._run_level(x, embs[-1], self.num_levels - 1)
        import pdb; pdb.set_trace()
        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, embs[i_level], i_level, is_up=True)
```

num parameteres: 458,818,051
shapes

```
original shape x: torch.Size([16, 128, 128, 128] ( b t c h w)

down sample i_level: 0, x.shape: torch.Size([16, 128, 128, 128])                                                                              
down sample i_level: 1, x.shape: torch.Size([16, 256, 64, 64])
down sample i_level: 2, x.shape: torch.Size([16, 576, 32, 32])

upsample i_level: 2, x.shape: torch.Size([16, 576, 32, 32])
upsample i_level: 1, x.shape: torch.Size([16, 256, 64, 64])
upsample i_level: 0, x.shape: torch.Size([16, 128, 128, 128])
```

## learnbale token

learnable token may need additional full finetuning

## Task

Finetune on longer context :

- first step: finetune on re10k with context 16 frame
- second step: finetune on re10k with vggt condition

看起来只要改一个uvit-pose里面的max length，然后rope embedding跟着变一下，然后去tune controlnet？

## prelimiary result

### re10 5000 video 5k step

- vggt-context16frame : fvd: 652
- dfot-16frame stage 1: fvd: 597
- stage2: dfot-vggt-resume: fvd: 636

## what about benchmark

num_videos_test:
training_horizon: 16
evaluation_context: 1

## blob backup source

v-haoywu/dfot-vggt-controlnet/
    - dfot-vggt-correctness-test-result-backup.tar: shows the crrectness of launching training
    - re10_dfot_download_plan.json: the url-timestpe dict from dfot download plan. to check alignment with others.

## Inference and evaluation benchmark

### 8 frames

```bash
python -m main +name=single_image_to_long dataset=realestate10k_mini algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous load=pretrained:DFoT_RE10K.ckpt 'experiment.tasks=[validation]' experiment.validation.data.shuffle=True dataset.context_length=1 dataset.frame_skip=1 dataset.n_frames=200 algorithm.tasks.prediction.keyframe_density=0.0625 algorithm.tasks.interpolation.max_batch_size=4 experiment.validation.batch_size=1 algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0 +algorithm.tasks.prediction.history_guidance.stabilization_level=0.02  algorithm.tasks.interpolation.history_guidance.name=vanilla +algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5
```

### 16 frames

change  checkepoint and max_frames in configurations/dataset_experiment/realestate10k_video_generation.yaml
all_videos in test_set is 200 frames.
keyframe_density set to 0.08

```bash
python -m main +name=single_image_to_long dataset=realestate10k_mini algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous load=checkpoints/DFoT_RE10K_16frame_5kft.ckpt 'experiment.tasks=[validation]' experiment.validation.data.shuffle=True dataset.context_length=1 dataset.frame_skip=1 dataset.n_frames=200 algorithm.tasks.prediction.keyframe_density=0.08 algorithm.tasks.interpolation.max_batch_size=4 experiment.validation.batch_size=1 algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0 +algorithm.tasks.prediction.history_guidance.stabilization_level=0.02  algorithm.tasks.interpolation.history_guidance.name=vanilla +algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5

```
