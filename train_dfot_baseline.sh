python -m main +name=RE10k dataset=realestate10k \
        algorithm=dfot_video_pose \
        experiment=video_generation @diffusion/continuous \
        load=pretrained:DFoT_RE10K.ckpt
