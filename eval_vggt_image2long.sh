
python -m main +name=single_image_to_long dataset=realestate10k_mini \
        algorithm=vggt_controlnet_dfot_video_pose experiment=video_generation \
        @diffusion/continuous load=checkpoints/DFoT_RE10K_16frame_5kft.ckpt \
        'experiment.tasks=[validation]' experiment.validation.data.shuffle=True \
        dataset.context_length=1 dataset.frame_skip=1 dataset.n_frames=200 \
        algorithm.tasks.prediction.keyframe_density=0.08 \
        algorithm.tasks.interpolation.max_batch_size=4 experiment.validation.batch_size=1 \
        algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla \
        +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0 \
        +algorithm.tasks.prediction.history_guidance.stabilization_level=0.02  \
        algorithm.tasks.interpolation.history_guidance.name=vanilla \
        +algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5 \
        'algorithm.logging.metrics=[fvd]' \
        hydra.run.dir=./outputs/evaluations/test_eval 