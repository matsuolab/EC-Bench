import json
import torch
import torchcodec
from transformers import TorchAoConfig, Qwen3VLForConditionalGeneration, AutoProcessor
import deepspeed

from .utils.utils import second2timestamp

class Qwen3VL:
    def __init__(self, args, config):
        model_path = config["model"]["model_name"]
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            # device_map="cuda",
        )
        self.model.eval()
        self.device = self.model.device

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            max_pixels=config["model"]["max_pixels"]
        )
        
        if config["deepspeed"]["use_deepspeed"]:
            self.model = deepspeed.init_inference(
                self.model,
                dtype=torch.bfloat16,
                replace_with_kernel_inject=True,
                max_tokens=config["deepspeed"]["max_tokens"],
                tensor_parallel={
                    "tp_size": args.num_gpus       # ここでTPの分割数を指定
                }
            )
        
        self.max_new_tokens = config["model"]["max_new_tokens"]
        self.num_frames = config["model"]["num_frames"]
        self.local_rank = args.local_rank

        with open(config["model"]["prompt_path"], "r") as f:
            self.prompt_base = f.read().strip()
        
        self.video_path = None
    
    def qa(self, query, video_path):
        frame_timestamps = self._get_timestamp(video_path)

        prompt = self.prompt_base.replace(
            "<QUESTION>", query
        ).replace(
            "<TIMESTAMPS>", str(self.frame_timestamps)
        )

        output_text = self.generate(prompt)
        if output_text.startswith("```json"):
            output_text = output_text.strip("```json").strip("```").strip()
        if output_text.startswith("{") and ("\"clip\"" in output_text) and output_text[-1] != "}":
            output_text = output_text.rsplit("]")[0] + "]]}"

        try:
            output_json = json.loads(output_text)
        except json.decoder.JSONDecodeError as jde:
            output_json = {
                "answer": output_text,
                "clip": []
            }
        except Exception as e:
            print(output_text)
            print(e)
            raise e
        
        output_json["clip"] = [
            c for c in output_json["clip"] if c[-1] <= self.frame_timestamps[-1]
        ]
        return output_json

    def generate(self, query):

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": self.video_path},
                    {"type": "text", "text": query},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            num_frames=self.num_frames,
            fps=None,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(f"cuda:{self.local_rank}")
        # import pdb; pdb.set_trace()

        # Inference: Generation of the output
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
    
    def _get_timestamp(self, video_path):

        if self.video_path == video_path:
            return self.frame_timestamps

        self.video_path = video_path

        decoder = torchcodec.decoders.VideoDecoder(video_path)
        metadata = decoder.metadata

        total_num_frames = metadata.num_frames
        original_fps = metadata.average_fps

        assert total_num_frames > 0
        assert original_fps > 0

        self.frame_timestamps = (torch.arange(0, total_num_frames, total_num_frames / self.num_frames)/original_fps).tolist()
        self.frame_timestamps = [int(round(x)) for x in self.frame_timestamps]
        self.frame_timestamps = [second2timestamp(x) for x in self.frame_timestamps]

        return self.frame_timestamps