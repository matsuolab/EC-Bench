import os
import json
from tqdm import tqdm
import importlib
import argparse
import yaml
import shutil
import datasets

def main(args):

    with open(f"{args.config_path}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    module_name = config["model"]["module_name"]
    VLM_module = importlib.import_module(f"models.{module_name}")
    VLM = getattr(VLM_module, module_name)
    model = VLM(args, config)

    shutil.copy(config["model"]["prompt_path"], os.path.join(args.output_dir, "prompt.txt"))

    dataset = datasets.load_dataset("vai-org/EC-Bench")

    video_path_base = "data/video_list/{}.mp4"
    output_list = []

    for sample in tqdm(dataset["train"], desc="data:", ncols=100):
        video_path = video_path_base.format(sample["video_url"].split("watch?v=")[-1])

        output = model.qa(sample["question"], video_path)

        output["question_id"] = sample["question_id"]
        output["question"] = sample["question"]
        output_list.append(output)

        with open(f"{args.output_dir}/vlm_output.json", "w") as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)
        
        if len(output_list) >= 50:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path to evaluate")
    parser.add_argument("--output_dir", help="output path")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu")
    parser.add_argument("--local_rank", type=int, default=0, help="device (automatically passed by deepspeed)")
    args = parser.parse_args()

    main(args)