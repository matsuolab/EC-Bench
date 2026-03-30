import json
import torch
import torchcodec
from transformers import TorchAoConfig, Qwen3VLForConditionalGeneration, AutoProcessor
import deepspeed

class Qwen3VL:
    def __init__(self):
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
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
            model_path
        )
        
        self.max_new_tokens = 128

    def generate(self, query):

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        # Inference: Generation of the output
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0].strip()