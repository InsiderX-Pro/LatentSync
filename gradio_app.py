import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime

CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed = 1247,
):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")  # Change the filename as needed

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed)

    try:
        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.", output_path)
        return output_path  # Ensure the output path is returned
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--temp_dir",
            "temp",
            "--enable_deepcache",
        ]
    )

css = """
.footer {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 80px;
  background-color: white;
}
"""

# Create Gradio interface
with gr.Blocks(title="唇形同步", css=css) as demo:
    gr.Markdown(
        """
    <h1 align="center">唇形同步</h1>
    """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="上传视频")
            audio_input = gr.Audio(label="上传音频", type="filepath")

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=1.5,
                    step=0.1,
                    label="引导尺度",
                    info="较高的值可以提高唇形同步精度，但可能会导致视频失真或抖动。",
                )
                inference_steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=20,
                    step=1,
                    label="推理步数",
                    info="值越高，视频质量越好，但生成速度越慢。",
                )

            process_btn = gr.Button("生成视频")

        with gr.Column():
            video_output = gr.Video(label="输出视频")

            # gr.Examples(
            #     examples=[
            #         ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
            #         ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
            #         ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
            #     ],
            #     inputs=[video_input, audio_input],
            # )
    gr.HTML(
        """
        <div class="footer"></div>
        """
    )

    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            inference_steps,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch(auth=("admin", "#SrDSic![fIl-|+?"), share=False)
