# Dfot-VGGT 

the original readme: dfot-readme.md 
dataset process: prepare-re10k-mp4.md 


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
- [ ] same initialization parametere
- [ ] same output from each downsampling blocks 
- [ ] zero-init
- [ ] forward 
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

看起来只要改一个uvit-pose里面的max length，然后rope embedding跟着变一下，然后去tune controlnet？ 

## what about benchmark 


## blob backup source: 

v-haoywu/dfot-vggt-controlnet/
    - dfot-vggt-correctness-test-result-backup.tar: shows the crrectness of launching training
    - re10_dfot_download_plan.json: the url-timestpe dict from dfot download plan. to check alignment with others. 

