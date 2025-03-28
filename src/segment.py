import os
import json
import shutil
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from sam2.build_sam import build_sam2_video_predictor

# --------------------------
# Visualization utility functions
# --------------------------
def show_mask(mask, ax, obj_id=None, random_color=False):
    """Visualize segmentation mask"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """Visualize interaction points"""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Visualize bounding box"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                               facecolor=(0, 0, 0, 0), lw=2))

def show_image_with_boxes(image, boxes, title="test"):
    """
    Display image with bounding boxes (merged version)
    :param image: Image array (H×W×3)
    :param boxes: List of bounding boxes, each element is [x0, y0, x1, y1]
    :param title: Image title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    for box in boxes:
        x0, y0, x1, y1 = box
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        plt.gca().add_patch(rect)
    ax.set_title(title)
    ax.axis('off')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    plt.show()

# --------------------------
# Core functionality module
# --------------------------
class VideoSegmentor:
    def __init__(self, device="cuda"):
        """
        Initialize video segmentor
        :param device: Computing device to use (cuda/cpu)
        """
        self.device = device
        self.predictor = None
        self.inference_state = None
        self.video_segments = {}
        self._model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self._checkpoint = "../checkpoints/sam2.1_hiera_large.pt"

    def load_model(self):
        """Load pre-trained model"""
        self.predictor = build_sam2_video_predictor(
            config_file=self._model_cfg,
            checkpoint_path=self._checkpoint,
            device=self.device
        )
        return self.predictor

    def init_video(self, video_dir):
        """
        Initialize video processing
        :param video_dir: Directory containing video frames
        """
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        self.frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        self.video_dir = video_dir
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        return self.inference_state

    def add_interaction(self, frame_idx, obj_id, points=None, labels=None, box=None):
        """
        Add user interaction (points/box)
        :param frame_idx: Interaction frame index
        :param obj_id: Target object ID
        :param points: Interaction points coordinates [[x,y],...]
        :param labels: Point labels (1=positive, 0=negative)
        :param box: Bounding box [x0,y0,x1,y1]
        """
        if points is not None and labels is not None:
            return self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=np.array(points, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32)
            )
        elif box is not None:
            return self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=np.array(box, dtype=np.float32)
            )
        else:
            raise ValueError("Must provide interaction points or bounding box")

    def add_point_prompt(self, frame_idx, obj_id, points, labels):
        """Add point prompt interaction"""
        return self.add_interaction(
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels
        )

    def add_box_prompt(self, frame_idx, obj_id, box):
        """Add bounding box prompt"""
        return self.add_interaction(
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box
        )

    def add_multi_box_prompts(self, frame_idx: list, obj_ids: list, boxes: list):
        """
        Add multiple object bounding box prompts for joint prediction
        :param frame_idx: Interaction frame index (same frame operation)
        :param obj_ids: Object ID list (must be unique and consecutive, e.g. [2,3])
        :param boxes: List of bounding boxes in [x_min,y_min,x_max,y_max] format
        """
        assert len(obj_ids) == len(boxes), "Object IDs and boxes count must match"
        assert len(obj_ids) == len(set(obj_ids)), "Object IDs must be unique"
        for frame_id, obj_id, box in zip(frame_idx, obj_ids, boxes):
            self.add_interaction(
                frame_idx=frame_id,
                obj_id=obj_id,
                box=np.array(box, dtype=np.float32)
            )
        return 1

    def reset_state(self):
        """Reset inference state"""
        if self.inference_state:
            self.predictor.reset_state(self.inference_state)
            self.video_segments = {}
            print("Inference state reset, ready for new segmentation process")

    @staticmethod
    def extract_frames(input_video="videos/bedroom.mp4",
                       output_dir="output/video_input",
                       quality=3,
                       start_number=0):
        """
        Video frame extraction
        :param input_video: Input video path
        :param output_dir: Output directory
        :param quality: JPEG quality (1-31)
        :param start_number: Starting frame number
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        output_pattern = os.path.join(output_dir, "%05d.jpg")
        command = [
            "ffmpeg",
            "-i", input_video,
            "-q:v", str(quality),
            "-start_number", str(start_number),
            output_pattern
        ]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully extracted video frames to {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Frame extraction failed: {e}")
        except FileNotFoundError:
            print("FFmpeg not found. Please ensure it's installed and in system PATH")

    def propagate(self):
        """Perform video propagation"""
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(out_obj_ids)
            }
        return self.video_segments

    def generate_combined_mask(self, frame_idx: int) -> np.ndarray:
        """
        Generate combined mask matrix with object IDs
        :param frame_idx: Target frame index
        :return: 2D array where 0=background, >0=object ID
        """
        if frame_idx not in self.video_segments:
            return np.zeros((1024, 1024), dtype=np.uint16)
        sample_mask = next(iter(self.video_segments[frame_idx].values())).squeeze()
        combined_mask = np.zeros_like(sample_mask, dtype=np.uint16)
        for obj_id, mask in self.video_segments[frame_idx].items():
            mask_2d = mask.squeeze().astype(bool)
            combined_mask[mask_2d] = obj_id
        return combined_mask

    def generate_all_combined_masks(self) -> dict:
        """
        Generate all frames' combined masks, skipping all-zero frames
        :return: Dictionary {frame_idx: combined_mask}
        """
        all_masks = {}
        for frame_idx, obj_masks in self.video_segments.items():
            sample_mask = next(iter(obj_masks.values())).squeeze()
            combined_mask = np.zeros_like(sample_mask, dtype=np.uint16)
            for obj_id, mask in obj_masks.items():
                mask_2d = mask.squeeze().astype(bool)
                combined_mask[mask_2d] = obj_id
            if np.any(combined_mask > 0):
                all_masks[frame_idx] = combined_mask
        return all_masks

    def analyze_results_statistics(self, verbose=False):
        """
        Analyze segmentation results
        :param verbose: Print detailed information
        :return: Analysis results dictionary
        """
        analysis = {
            "total_frames": len(self.video_segments),
            "frame_details": {}
        }
        for frame_idx, masks in self.video_segments.items():
            analysis["frame_details"][frame_idx] = {
                "object_count": len(masks),
                "objects": {obj_id: mask.shape for obj_id, mask in masks.items()}
            }
        if verbose:
            print(f"Total frames: {analysis['total_frames']}")
            for frame_idx, detail in analysis["frame_details"].items():
                print(f"Frame {frame_idx}:")
                print(f"  Object count: {detail['object_count']}")
                for obj_id, shape in detail['objects'].items():
                    print(f"    Object {obj_id}: Mask shape {shape}")
        return analysis

    def visualize_segmentation_results(self, vis_frame_stride=30):
        """
        Visualize segmentation results at specified frame intervals
        :param vis_frame_stride: Frame interval step size
        """
        plt.close("all")
        for out_frame_idx in range(0, len(self.frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"Frame {out_frame_idx}")
            frame_path = os.path.join(self.video_dir, self.frame_names[out_frame_idx])
            plt.imshow(Image.open(frame_path))
            if out_frame_idx in self.video_segments:
                for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.show()

# --------------------------
# Video segmentation pipeline
# --------------------------
def video_seg_pipeline(input_video, output_dir="output/video_input", quality=3, start_number=0):
    """
    Video segmentation processing pipeline
    :param input_video: Input video path
    :param output_dir: Output frame directory
    :param quality: JPEG quality (1-31)
    :param start_number: Starting frame number
    """
    segmentor = VideoSegmentor(device="cuda")
    segmentor.load_model()
    segmentor.init_video(output_dir)
    segmentor.add_point_prompt(
        frame_idx=0,
        obj_id=1,
        points=[[210, 350]],
        labels=[1])
    segmentor.add_multi_box_prompts(
        frame_idx=0,
        obj_ids=[2, 3],
        boxes=[
            [200, 250, 350, 400],
            [400, 100, 550, 300]
        ]
    )
    results = segmentor.propagate()
    ma = segmentor.generate_combined_mask(1)
    plt.imshow(ma)
    plt.show()
    all_masks = segmentor.generate_all_combined_masks()
    print(f"Length of all_masks: {len(all_masks)}")
    segmentor.visualize_segmentation_results(vis_frame_stride=20)
    return results

# --------------------------
# Main program
# --------------------------
if __name__ == "__main__":
    input_video = "videos/render_animation_micro.mp4"
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "config",
        "3d_render_video.json"
    )
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    results = video_seg_pipeline(input_video)