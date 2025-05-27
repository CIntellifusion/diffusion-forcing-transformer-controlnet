# Align Re10k from  PixelSplat Format to DFot Format

Diffusion forcing transformer requires mp4 format for trianing while in pixel splat is stored in binary. 

## Process in DFot Preparation. 

in datasets/video/realestate10.py there is a downlaod dataset function: 
1. download metadata(pose txt files )
2. contruct download plan of video from youtube 
3. preprocess videos into 256 resolution. 

### a download plan example 
the download plan is stored to re10_dfot_download_plan.json. in the following format: 
{
    url:[video_id:List[timesteps]]
}
each original video from youtube was split to several short videos according to timesteps. (as long as I have confirmed, the timesteps are consecutive.)

we checked the case 'https://www.youtube.com/watch?v=I_y_2Bwp6Bg' -> '4ceba2c95ef00194' from 1:49 to 1:58 about 10 seconds. 

### preprocess 

after download original videos, they will be process in _preprocess_video(datasets/video/realestate10.py) by 
1. load the corressponding video frames of each timesteps. 
2. rescale and crop to target resolution. 


# The organization of pixelsplat 

pixel splat has a video foramt of .torch files. each torch files contains serveral items processed. (refer https://github.com/dcharatan/pixelsplat/blob/main/src/dataset/dataset_re10k.py#L202 for more details)

## Structure of .torch files 

torch.load (xxx.torch) then we will have a structure like this : 
[
    {'url', 'timestamps', 'cameras', 'images', 'key'},

]

The relationship between two processing pipeline is : 
- pixelsplat  dfot meaning
- url  url url_on_youtube 
- timesteps timesteps index_in_original_video 
- images video_clips_by_timestep in original size(360 640)
- key video_id the_clip_id_for_training_

# To process pixelsplat into dfot format: 
- load the .torch files 
- read the videos clips 
- resize and crop 
- save 
