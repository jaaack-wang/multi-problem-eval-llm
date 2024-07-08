import os
import argparse
import pandas as pd
from tqdm import tqdm
from scripts.llm import *
from scripts.utils import *


def experiment(input_fp, model, save_freq=50, max_tokens=None, overwrite=False, api_key=None):        
    data = pd.read_json(input_fp, lines=True)
    
    if f"{model}-completion" in data.columns and not overwrite:
        sub = data[data[f"{model}-completion"].notnull()]
        indices = sorted(set(data.index) - set(sub.index))
    else:
        indices = data.index

    if len(indices) == 0:
        print(f"All completions already done!")
        return

    counter = 0
    for ix in tqdm(indices):
        if counter % save_freq == 0:
            data.to_json(input_fp, orient='records', lines=True)

        prompt = data.loc[ix, 'prompt']
        completion = get_completion(prompt, model, max_tokens=max_tokens, api_key=api_key)

        data.at[ix, f"{model}-completion"] = completion
        counter += 1

    data.to_json(input_fp, orient='records', lines=True)
    print(f"{input_fp} updated with {model} completions!")


def bool_eval(name, boolean):
    if isinstance(boolean, bool):
        return boolean
    
    boolean = boolean.lower()
    if boolean == "true":
        return True
    if boolean == "false":
        return False
    raise TypeError(f"{name} must be boolean, but {type(boolean)} was given.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125", help="model card name. Defaults to gpt-3.5-turbo-0125.")
    parser.add_argument("--datasets", type=str, default="AGNews", help="datasets to use, separated by comma. Defaults to AGNews.")
    parser.add_argument("--save_freq", type=int, default=10, help="frequency to save completions every save_freq completions. Defaults to 10.")
    parser.add_argument("--max_tokens", type=int, default=None, help="max number of output tokens for a model completion. Defaults to None.")
    parser.add_argument("--overwrite", type=str, default=False, help="whether to overwrite previous model completions. Defaults to False.")
    parser.add_argument("--api_key", type=str, default=None, help="api_key if you do not want to use default. Defaults to None.")
    
    args = parser.parse_args()
    print('*****************************')
    print(args)
    print('*****************************')
    
    model = args.model
    datasets = [d.strip() for d in args.datasets.split(",")]
    overwrite = bool_eval("overwrite", args.overwrite)

    for d in datasets:

        input_fp = f"results/text classification/{d}.json"

        if os.path.exists(input_fp):
            print(f"Doing experiments with {input_fp}")
            experiment(input_fp, model, save_freq=args.save_freq,
                       max_tokens=args.max_tokens, overwrite=overwrite, api_key=args.api_key)
        else:
            print(input_fp + " does not exist.")
