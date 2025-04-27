import argparse
import json
from open_r1.utils import get_tokenizer, get_model
from datasets import load_dataset
from tqdm import tqdm

OOD = {
    "math":        ("open_r1.data.combined_loader", "math", "test"),
    "riddlesense": ("open_r1.data.combined_loader", "riddlesense", "validation"),
    "brainteaser": ("open_r1.data.combined_loader", "brainteaser", "train[:1000]"),
}

def accuracy(model, tok, dataset):
    right = 0
    for ex in tqdm(dataset, desc="eval"):
        prompt = ex["prompt"] + "\n<answer>\n"
        out = model.generate(**tok(prompt, return_tensors="pt").to("cuda"), max_new_tokens=32)
        pred = tok.decode(out[0], skip_special_tokens=True).split("<answer>")[-1].strip().split()[0]
        right += pred == str(ex["solution"]).split()[0]
    return right / len(dataset)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")                 # path like checkpoints/gsm8k_only/checkpoint-0001
    args = p.parse_args()

    tok = get_tokenizer(None, None, model_name_or_path=args.checkpoint)
    model = get_model(None, None, model_name_or_path=args.checkpoint).eval().cuda()

    scores = {}
    for name, (dname, cfg, split) in OOD.items():
        ds = load_dataset(dname, cfg, split=split)
        scores[name] = accuracy(model, tok, ds)

    print(json.dumps(scores, indent=2)) 