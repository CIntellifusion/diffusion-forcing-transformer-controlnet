# python -m main +name=RE10k dataset=realestate10k \
#         algorithm=vggt_controlnet_dfot_video_pose \
#         experiment=video_generation @diffusion/continuous \
#         load=pretrained:DFoT_RE10K.ckpt

python -m main +name=RE10k dataset=realestate10k \
        algorithm=vggt_controlnet_dfot_video_pose \
        experiment=video_generation @diffusion/continuous \
        load=checkpoints/DFoT_RE10K_16frame_5kft.ckpt
        