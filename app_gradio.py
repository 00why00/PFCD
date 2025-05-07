import argparse
import multiprocessing as mp
import os
import random
import time

import torch

from pipelines.pipeline_stable_xl_image_variation import StableDiffusionXLImageVariationPipeline


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Launch gradio application")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./inference/saved_pipeline/stable-xl-image-variation",
        help="pretrained model path",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Model checkpoint"
    )
    parser.add_argument(
        "--device",
        nargs="+",
        type=int,
        default=[0],
        help="Index of devices"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9919,
        help="Server port"
    )
    return parser.parse_args()


class InferenceCommand:
    """Command to run batched inference."""

    def __init__(self, input_queue, output_queue, kwargs):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.kwargs = kwargs

    def _build_env(self):
        """Build the environment."""
        self.batch_size = self.kwargs.get("batch_size", 1)
        self.batch_timeout = self.kwargs.get("batch_timeout", None)

    def build_pipeline(self):
        """Build and return the model."""
        pipeline = StableDiffusionXLImageVariationPipeline.from_pretrained(
            self.kwargs["pretrained_model_name_or_path"],
            torch_dtype=torch.float16,
            local_files_only=True
        )
        pipeline.load_lora_weights(self.kwargs["checkpoint_path"])
        pipeline = pipeline.to(self.kwargs["device"])
        return pipeline

    def send_results(self, pipeline, indices, examples):
        """Send the inference results."""
        for i, example in enumerate(examples):
            result = pipeline(
                    image=example["img"],
                    height=512,
                    width=512,
                    generator=torch.Generator(self.kwargs["device"]).manual_seed(example["seed"]),
                    use_content=True,
                    image_info=example["image_info"],
                ).images[0]
            self.output_queue.put((indices[i], result))

    def run(self):
        """Main loop to make the inference outputs."""
        self._build_env()
        pipeline = self.build_pipeline()
        must_stop = False
        while not must_stop:
            indices, examples = [], []
            deadline, timeout = None, None
            for i in range(self.batch_size):
                if self.batch_timeout and i == 1:
                    deadline = time.monotonic() + self.batch_timeout
                if self.batch_timeout and i >= 1:
                    timeout = deadline - time.monotonic()
                try:
                    index, example = self.input_queue.get(timeout=timeout)
                    if index < 0:
                        must_stop = True
                        break
                    indices.append(index)
                    examples.append(example)
                except Exception:
                    pass
            if len(examples) == 0:
                continue
            self.send_results(pipeline, indices, examples)


class ServingCommand(object):
    """Command to run serving."""

    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.output_dict = mp.Manager().dict()
        self.output_index = mp.Value("i", 0)

    def run(self):
        """Main loop to make the serving outputs."""
        while True:
            img_id, outputs = self.output_queue.get()
            self.output_dict[img_id] = outputs


def build_gradio_app(queues, command):
    """Build the gradio application."""
    import gradio as gr
    import gradio_image_prompter as gr_ext

    title = "Multi-object Image Augmentation"
    header = (
        "<div align='center'>"
        "<h1>Prompt-Free Conditional Diffusion for Multi-object Image Augmentation</h1>"
        "<h3><a href='https://arxiv.org/abs/xxxx.xxxxx' target='_blank' rel='noopener'>[paper]</a>"
        "<a href='https://github.com/00why00' target='_blank' rel='noopener'>[code]</a></h3>"  # noqa
        "</div>"
    )
    theme = "soft"
    css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
             #anno-img .mask.active {opacity: 0.7}"""

    def get_click_examples():
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        app_images = list(filter(lambda x: x.startswith("app_image"), os.listdir(assets_dir)))
        app_images.sort()
        return [{"image": os.path.join(assets_dir, x)} for x in app_images]

    def on_reset_btn():
        click_img, draw_img = gr.Image(None), gr.ImageEditor(None)
        anno_img = gr.AnnotatedImage(None)
        return click_img, draw_img, anno_img

    def on_submit_btn(click_img, seed_opt):
        if seed_opt == -1:
            seed = random.randint(0, 2 ** 32)
        else:
            seed = seed_opt
        assert click_img is not None
        img, points = click_img["image"], click_img["points"]
        img = img.convert("RGB")
        instances = []
        for point in points:
            if point[2] == 2.0 and point[5] == 3.0:
                instances.append({"bbox": [point[0], point[1], point[3] - point[0], point[4] - point[1]]})
        inputs = {
            "img": img,
            "seed": seed,
            "image_info": {
                "image": img,
                "height": img.height,
                "width": img.width,
                "instances": instances,
            }
        }
        with command.output_index.get_lock():
            command.output_index.value += 1
            img_id = command.output_index.value
        queues[img_id % len(queues)].put((img_id, inputs))
        while img_id not in command.output_dict:
            time.sleep(0.005)
        outputs = command.output_dict.pop(img_id)
        return outputs

    app, _ = gr.Blocks(title=title, theme=theme, css=css).__enter__(), gr.Markdown(header)
    container, column = gr.Row().__enter__(), gr.Column().__enter__()
    click_img = gr_ext.ImagePrompter(show_label=False, type="pil")
    gr.Markdown("<h3 style='text-align: center'>PressMove to draw bounding boxes</h3>")
    seed_opt = gr.Number(label="Seed", value=-1, scale=1, info="-1 for random seed.")
    gr.Examples(get_click_examples(), inputs=[click_img])
    row, reset_btn, submit_btn = gr.Row().__enter__(), gr.Button("Reset"), gr.Button("Execute")
    _, _, column = row.__exit__(), column.__exit__(), gr.Column().__enter__()
    output_img = gr.Image(elem_id="output-img", show_label=False)
    reset_btn.click(on_reset_btn, [], [click_img, output_img])
    submit_btn.click(on_submit_btn, [click_img, seed_opt], [output_img])
    column.__exit__(), container.__exit__(), app.__exit__()
    return app


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    queues = [mp.Queue(1024) for _ in range(len(args.device) + 1)]
    commands = [
        InferenceCommand(
            queues[i],
            queues[-1],
            kwargs={
                "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "checkpoint_path": args.checkpoint_path,
                "device": args.device[i],
                "verbose": i == 0,
            },
        )
        for i in range(len(args.device))
    ]
    commands += [ServingCommand(queues[-1])]
    actors = [mp.Process(target=command.run, daemon=True) for command in commands]
    for actor in actors:
        actor.start()
    app = build_gradio_app(queues[:-1], commands[-1])
    app.queue()
    app.launch(server_port=args.port, share=False)