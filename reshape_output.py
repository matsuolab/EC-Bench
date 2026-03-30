import os
import json
import argparse
from tqdm import tqdm

from models.Qwen3VL_for_reshape import Qwen3VL

def main(args):

    with open(os.path.join(args.output_dir, "vlm_output.json"), "r") as f:
        vlm_output = json.load(f)

    with open("prompts/reshape.txt", "r") as f:
        base_prompt = f.read().strip()
    
    model = Qwen3VL()

    reshaped_output = []

    for res in tqdm(vlm_output):
        prompt = base_prompt.replace(
            "<QESTION>", res["question"]
        ).replace(
            "<ANSWER>", str(res["answer"])
        )
        reshaped_answer = model.generate(prompt)
        reshaped_output.append(
            {
                "question_id": res["question_id"],
                "answer": reshaped_answer,
                "clip": res["clip"]
            }
        )
    
    with open(os.path.join(args.output_dir, "reshaped_output.json"), "w") as f:
        json.dump(reshaped_output, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="output path")
    args = parser.parse_args()

    main(args)