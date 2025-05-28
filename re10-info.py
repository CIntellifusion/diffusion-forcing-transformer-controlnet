path = './data/real-estate-10k/'
import os

test_video_path =os.path.join(path, 'test_256')
train_video_path = os.path.join(path, 'training_256')
test_pose_path = os.path.join(path, 'test_poses')
train_pose_path = os.path.join(path, 'training_poses')

def filename2videoid(filename):
    """
    Convert filename to video ID.
    Example: '000001.mp4' -> '000001'
    """
    return os.path.splitext(filename)[0]

train_video_id = [filename2videoid(f) for f in os.listdir(train_video_path) if f.endswith('.mp4')]
test_video_id = [filename2videoid(f) for f in os.listdir(test_video_path) if f.endswith('.mp4')]
train_pose_id = [filename2videoid(f) for f in os.listdir(train_pose_path) if f.endswith('.pt')]
test_pose_id = [filename2videoid(f) for f in os.listdir(test_pose_path) if f.endswith('.pt')]
print(f"dataset info: real-estate-10k from {path}")
print(f"Train videos: {len(train_video_id)}")
print(f"Test videos: {len(test_video_id)}")
print(f"Train poses: {len(train_pose_id)}")
print(f"Test poses: {len(test_pose_id)}")

# how many ids not in train_video_id but in train_pose_id
train_not_in_video = set(train_pose_id) - set(train_video_id)
print(f"Train poses not in videos: {len(train_not_in_video)}")
# how many ids not in test_video_id but in test_pose_id
test_not_in_video = set(test_pose_id) - set(test_video_id)
print(f"Test poses not in videos: {len(test_not_in_video)}")

# how many ids in train_video_id but not in train_pose_id
train_not_in_pose = set(train_video_id) - set(train_pose_id)
print(f"Train videos not in poses: {len(train_not_in_pose)}")
# how many ids in test_video_id but not in test_pose_id
test_not_in_pose = set(test_video_id) - set(test_pose_id)

print(f"Test videos not in poses: {len(test_not_in_pose)}")

# how many ids in train video id and in test video id
print(f"Train and test video ids: {len(set(train_video_id) & set(test_video_id))}")
# how many ids in train pose id and in test pose id
print(f"Train and test pose ids: {len(set(train_pose_id) & set(test_pose_id))}")
# training videos in test poses 
print(f"Training videos in test poses: {len(set(train_video_id) & set(test_pose_id))}")
print(f"test videos in train poses: {len(set(test_video_id) & set(train_pose_id))}")
