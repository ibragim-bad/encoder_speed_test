import click
import time
import yaml
import pandas as pd

import numpy as np

N_CPU = 8

CPU = str(N_CPU)
MKL = str(N_CPU + 2)

import os

os.environ["OMP_NUM_THREADS"] = CPU  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = CPU  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = MKL  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU  # export VECLIB_MAXIMUM_THREADS=4
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["NUMEXPR_NUM_THREADS"] = CPU  # export NUMEXPR_NUM_THREADS=6

import torch

torch.set_num_threads(N_CPU)

from transformers import AutoTokenizer, AutoModel

import copy


def make_kwargs_combinations(kwargs):
    def _update_and_copy(dict1: dict, dict2: dict) -> dict:
        dict1_copy = copy.deepcopy(dict1)
        dict1_copy.update(dict2)
        return dict1_copy

    kwargs_combinations = [{}]
    for key, value in kwargs.items():
        if not isinstance(value, list):
            for kwargs_combination in kwargs_combinations:
                kwargs_combination[key] = value
            continue

        kwargs_combinations = [
            _update_and_copy(kwargs_combination, {key: item})
            for item in value
            for kwargs_combination in kwargs_combinations
        ]

    return kwargs_combinations


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def run_model(
    texts, model, tokenizer: AutoTokenizer, bs, only_tokenize, cut, max_length
):
    embs = []
    texts = [t[:cut] for t in texts]
    for bt in batch(texts, n=bs):
        batch_dict = tokenizer.batch_encode_plus(
            bt,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        if not only_tokenize:
            outputs = model(**batch_dict)
            embeddings = average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

            embs.append(embeddings)
    # if not only_tokenize:
    #     embs = torch.vstack(embs)
    return embs


@click.command()
@click.option("--config", "-c", type=str, required=True)
def main(
    config: str,
):
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    with open(config["input"], "r") as f:
        text = f.read()

    upside = 100_000 // len(text) + 1
    text = " ".join(text * upside)
    docs = [text] * config["docs_count"]

    outputs = []

    for model_name in config["models"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        loaded = AutoModel.from_pretrained(model_name)
        params = make_kwargs_combinations(config["params"])

        loaded(
            **tokenizer.batch_encode_plus(
                [text],
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
        )
        print(f"{model_name} warmed")

        for param_combo in params:
            times = []
            for _ in range(config["num_runs"]):
                st = time.monotonic()
                embs = run_model(
                    docs,
                    loaded,
                    tokenizer,
                    param_combo["batch_size"],
                    only_tokenize=False,
                    cut=param_combo["clip_str"],
                    max_length=param_combo["seq_len"],
                )
                tt = time.monotonic() - st
                times.append(1000 * tt)

            times = np.array(times)
            print(param_combo)
            print(
                f"{times.mean():.3f} per sample: {times.mean()/len(docs):.3f} ms, std_prc of runs: {times.std()/times.mean():.3f}"
            )
            outputs.append(
                {
                    "model": model_name,
                    "one_doc_ms": round(times.mean() / len(docs), 3),
                    "mean_sum_ms": round(times.mean(), 3),
                    "std_prc": round(times.std() / times.mean(), 3),
                    **param_combo,
                }
            )

    df = pd.DataFrame(outputs)
    df.to_csv(config["output"], index=False, sep="\t")


if __name__ == "__main__":
    main()
