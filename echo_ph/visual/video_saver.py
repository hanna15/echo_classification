import cv2
import os


class VideoSaver():
    def __init__(self, video_id, grad_cam_frames, out_dir='vis_videos', max_frames=50, fps=10):
        self.video_id = str(video_id)
        self.grad_cam_frames = grad_cam_frames
        self.max_frames = max_frames
        self.fps = fps
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save_video(self):
        size = (self.grad_cam_frames[0].shape[0], self.grad_cam_frames[0].shape[1])
        out_path = os.path.join(self.out_dir, self.video_id + '.avi')
        result = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, size)
        frames = self.grad_cam_frames[0:self.max_frames]  # E.g. only first 50 frames, or approx first 5 heart-beats
        for frame in frames:
            result.write(frame)  # Write the frame into the file .avi file
        result.release()
        cv2.destroyAllWindows()  # Closes all the frames

