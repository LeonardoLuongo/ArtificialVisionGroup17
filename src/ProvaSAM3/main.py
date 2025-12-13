import torch
from sam3.model_builder import build_sam3_video_predictor
from config import HUGGINGFACE_API_TOKEN

from huggingface_hub import login
login(HUGGINGFACE_API_TOKEN)

video_predictor = build_sam3_video_predictor()
video_path = "./dataset/tracking-2023/train/SNMOT-060/img1" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="Players",
    )
)
output = response["outputs"]
