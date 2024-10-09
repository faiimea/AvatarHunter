# # Scheme 1: use ffmpeg (maybe can only support running locally)
# import subprocess

# input_video = "record.mp4"
# output_prefix = "output_video_"

# # Get video duration using FFmpeg
# ffprobe_cmd = f'ffprobe -i {input_video} -show_entries format=duration -v quiet -of csv="p=0"'
# duration = float(subprocess.check_output(ffprobe_cmd, shell=True))

# # Cut video into 1-second segments
# for i in range(int(duration // 1)):
#     start_time = i * 1
#     output_video = f"{output_prefix}{i}.mp4"
#     ffmpeg_cmd = f'ffmpeg -ss {start_time} -i {input_video} -t 1 -an -c:v copy {output_video}'
#     subprocess.call(ffmpeg_cmd, shell=True)
# # An important fact is, not all video have same video rate.
# # So we need to modify argument according to certain video



# # Scheme 2: use torch
# import torch
# import torchvision.io as io

# # Read video file
# video, _, info = io.read_video('record.mp4')

# # Get video properties
# fps = info['video_fps']
# frame_count = info['video_frames']
# frame_width = info['video_width']
# frame_height = info['video_height']

# # Cut video into 8-frame segments
# for i in range(0, frame_count, 8):
#     output_video = f"output_video_{i}.pt"
#     video_frames = video[i:i+8]
#     torch.save(video_frames, output_video)

# Scheme 3: use opencv
import cv2
import numpy as np

# Read video file using OpenCV
cap = cv2.VideoCapture('output_video_144.mp4')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Cut video into 8-frame segments
for i in range(0, frame_count, 8):
    # Read 8 frames from video
    frames = []
    for j in range(8):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    # Convert frames to numpy array
    video_frames = np.array(frames)

    # Save video frames to numpy file
    j = int(i/8) + 1 
    output_video = f"output_video_p{j}.npy"
    
    if not ret:
        break
    np.save(output_video, video_frames)
    

# Release video capture object
cap.release()
    
test = np.load('output_video_p0.npy')

print(test)

# 4. Test if .npy can be converted into .mp4

import cv2
import numpy as np
# Load video frames from numpy file
video_frames = np.load('output_video_p0.npy')

# Get video properties
frame_count, frame_height, frame_width, channels = video_frames.shape
fps = 30  # Set the frame rate

# Initialize a new video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Set the video codec
out = cv2.VideoWriter('output_video_test.mp4', fourcc, fps, (frame_width, frame_height))

# Write video frames to output file
for i in range(frame_count):
    frame = video_frames[i]
    out.write(frame)

# Release video writer object
out.release()


# # Load video frames from numpy file
# video_frames = np.load('output_video_72.npy')

# # Get video properties
# frame_count, frame_height, frame_width, channels = video_frames.shape
# fps = 30  # Set the frame rate

# # Initialize a new video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Set the video codec
# out = cv2.VideoWriter('output_video_test.mp4', fourcc, fps, (frame_width, frame_height))

# # Write video frames to output file
# for i in range(frame_count):
#     out.write(video_frames[i])

# # Release video writer object
# out.release()































