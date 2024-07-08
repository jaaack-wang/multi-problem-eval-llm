import os
import re
import json
import pandas as pd


benchmarks_gt_labels = {}

for benchmark in os.listdir("data/databases/text classification/"):
    if not benchmark.endswith(".pkl"):
        continue
    name = benchmark.replace(".pkl", "")
    benchmarks_gt_labels[name] = pd.read_pickle(f"data/databases/text classification/{benchmark}")["labels"]

benchmarks_gt_labels["AGNews-simplified_index_selection_only"] = benchmarks_gt_labels["AGNews"]
benchmarks_gt_labels["CoLA_2_shot"] = benchmarks_gt_labels["CoLA"]

RECOGNIZED_BENCHMARKS = set(benchmarks_gt_labels.keys())


def extract_answer(completion, task, benchmark, model=None, taskSize=None, 
                   targetLabel=None, print_unparsable=False, SelectOne_json_output=True):
    if not isinstance(completion, str):
        return "CANNOT_PARSE"

    completion_copy = completion
    assert benchmark in RECOGNIZED_BENCHMARKS, f"Unrecognized benchmark: {benchmark}. " \
                                                f"Recognized benchmarks: {list(benchmarks_gt_labels.keys())}"
    labels = benchmarks_gt_labels[benchmark]

    if task != "SingleClf":
        assert taskSize is not None, "taskSize must be provided for tasks other than SingleClf"

    if task == "SelectOne":
        assert targetLabel is not None, "targetLabel must be provided for SelectOne tasks" 

    if task == "SingleClf":
        for flags in [0, re.I]: # fistr strict (case-sentiive), then case-insensitive
            target = re.search(r"\b(" + "|".join(labels) + r")\b", completion, flags=flags)
            if target:
                return target.group()
        
        # very conservative rules to deal with mistralai/Mistral-7B-Instruct-v0.2 not giving direct labels
        if model == "mistralai/Mistral-7B-Instruct-v0.2" and benchmark=="WiC":
            first_sentence = completion.split(".")[0].lower()
            if "is different" in first_sentence or "not the same" in first_sentence:
                return "No"
            elif "different" in first_sentence and "not" not in first_sentence and "same" not in first_sentence:
                return "No"
            elif ("same" in first_sentence or "consistent" in first_sentence) and "not" not in first_sentence:
                return "Yes"
        
        if model == "mistralai/Mistral-7B-Instruct-v0.2" and benchmark=="SNLI":
            completion = completion.lower()
            # Neutral will not exist; otherwise it will be extracted already
            if ("entail" in completion or "identical" in completion) and "contradit" not in completion: 
                return "Entailment"
            # if "entail" or "identical" in completion, the elif statement will be not evaluated
            elif "contradit" in completion:
                return "Contradiction"

        if model == "mistralai/Mistral-7B-Instruct-v0.2" and benchmark=="MRPC":
            # when not appear before "paraphrase" or "identical", it means not a paraphrase
            if re.search(r"\bnot[^.]*(paraphrase|identical)", completion, flags=re.I):
                return "No"
            if "is a paraphrase of" in completion.lower():
                return "Yes"

        if model == "mistralai/Mistral-7B-Instruct-v0.2" and benchmark=="SST-2-inference":
            # do not use "flags=re.I" here to ensure the captured pattern appears in one sentence  
            if not re.search(r"\b(not possible to determine|impossible to determine|unable to determine|difficult to determine|cannot determine|not directly comparable)\b", completion):
                if re.search(r"\bsentiments?.*not\b.* same", completion):
                    return "No"
                elif re.search(r"\bsentiments?.* different from", completion):
                    return "No"
                elif re.search(r"\bdoe?s? not share the same sentiment", completion):
                    return "No"
                elif re.search(r"\bshares the same sentiment", completion):
                    return "Yes"
    
    elif task == "BatchClf":
        # for sentiment analysis, "Neutral" and "Mixed" are found to be frequently predicted labels
        # important to include them here to ensure the order of predicted labels are correct
        if benchmark in {"SST-2"}:
            labels = labels + ["Neutral", "Mixed"]
        
        # e.g., labels line by line without any numbering
        for flags in [0, re.I]:
            targets = re.findall(r"\b(" + "|".join(labels) + r")\b", completion, flags=flags)
            if len(targets) == taskSize:
                return targets
        
        for flags in [0, re.I]:
            # label preceded by ":", e.g., Answer: Label; Answer: Label; Answer: Label, etc.
            answered_targets = re.findall(r": ?(" + "|".join(labels) + r")\b", completion, flags=flags)
            answered_targets = [re.sub(r": ?", "", target) for target in answered_targets]
            if len(answered_targets) == taskSize:
                return answered_targets

        # the model may do classification line by line and some lines may have unrecognized labels
        # adding "\n" to the two sides of completion in case the completion starts/end with the index
        indexed_texts = re.split(r"\n\d+\. ?", "\n" + completion.strip() + "\n", flags=re.I)[1:] 

        # if there are as many lines as the questions, treat each line as a SingleClf problem 
        if len(indexed_texts) == taskSize:
            return [extract_answer(text, "SingleClf", benchmark, model, None, False) for text in indexed_texts]
        
        # if none of the above works, just return targets which capture all the desired labels
        if len(targets) != 0:
            return targets
    
    elif task == "SelectOne" and not SelectOne_json_output:
        completion = "\n" + completion.strip() + "\n"
        completion = re.sub(r"\n+", "\n", completion) # remove redundant lines 

        if benchmark == "WiC": # the prompts for WiC contain "Context 1/2" that may be repeated by some models
            completion = re.sub(r"Context [12]", "", completion, flags=re.I)
        
        # remove the explanation part
        if len(re.findall(r"explanation[^:]*:", completion, flags=re.I)) == 1:
            completion = re.sub(r"explanation[^:]*:[\S\s]+", "", completion, flags=re.I)
        
        # some models may summarize the final answers. here, only consider cases where "answer/output...:" followed by a number
        if len(re.findall(r"answer[^:]*:\n? ?\d", completion, flags=re.I)) == 1:
            completion = re.search(r"answer[^:]*:[\S\s]+", completion, flags=re.I).group()

        elif len(re.findall(r"output[^:]*:\n? ?\d", completion, flags=re.I)) == 1:
            completion = re.search(r"output[^:]*:[\S\s]+", completion, flags=re.I).group()
        
        # remove the self-negated parts: mostly found for meta-llama/Llama-3-70b-chat-hf answering questions from SST-2 & CoLA,
        # the corection is a disclaimer in the form of "(wait no)", "(... it is a mistake)", etc. in the end of each line
        if model == "meta-llama/Llama-3-70b-chat-hf":
            if benchmark == "SST-2":
                completion = re.sub(r"\d+\.?.*\(.*\bno\b.*\)", "", completion)
                completion = re.sub(r"\d+\.?.*\(.*mistake.*\)", "", completion)
                completion = re.sub(r"\d+\.?.*\(.*remove.*\)", "", completion)
            elif benchmark == "CoLA" and targetLabel.lower() == "unacceptable":
                completion = re.sub(r"\d+\.?.*\(.*no error.*\)", "", completion) 
        
        # Use capitalized None/All as required by the prompts and avoid matching "none/all" in the text
        choices = [str(i) for i in range(1, taskSize + 1)] + ["None", "All"]

        # some models may produce consecutive indexed answers, e.g., 1. 4; 2. 6 where the real intended answers are 4 and 6
        indexed_answers = re.findall(r"\d+\. \d+\.?", completion)
        if all([indexed_answers[i].split(".")[0] == str(i+1) for i in range(len(indexed_answers))]):
            completion = re.sub(r"(\d+\. )(\d+\.?)", r"\2", completion)

        # must be a list of numbers separated by commas: this is a rare case, but need to be considered
        targets_in_a_row = []
        nums_in_a_row = re.search(r"\d+,[\d, ]+", completion)
        if nums_in_a_row:
            targets_in_a_row = re.findall(r"\d+", nums_in_a_row.group())

        targets_line_by_line = re.findall(r"(?<=\n)(" + "|".join(choices) + r")\b", completion)

        # take whatever captures more indices
        if len(set(targets_line_by_line)) < len(set(targets_in_a_row)):
            targets = targets_in_a_row
        else:
            targets = targets_line_by_line
 
        if "All" in targets:
            if len(targets) == 1 or len(set(targets))-1 == taskSize:
                targets = ["All"]
            else: # remove "All" that does not mean all the possible indices
                targets = [t for t in targets if t != "All"]

        if "None" in targets:
            if len(set(targets)) != 1:
                targets = [t for t in targets if t != "None"]
            else:
                targets = ["None"]

        # if all the targets are unique, just return them
        if len(targets) != 0 and len(targets) == len(set(targets)):
            return [int(t) if t.isdigit() else t for t in targets]

        # sometime models may analyze all texts line by line before producing an answer
        if len(targets) > taskSize:
            if all([targets[i] == str(i+1) for i in range(taskSize)]):
                return [int(t) if t.isdigit() else t for t in targets[taskSize:]]
            elif all([targets[-taskSize:][i] == str(i+1) for i in range(taskSize)]):
                return [int(t) if t.isdigit() else t for t in targets[:-taskSize]]

        # the models may analyze all the indices and then provide the answer
        nums = re.findall(r"\d+", completion)
        if len(nums) > taskSize and all([nums[i] == str(i+1) for i in range(taskSize)]):
            return [int(n) if n.isdigit() else n for n in nums[taskSize:]]

        # if none of the above works, just return whatever nums that appear in the text
        if len(nums) > 0:
            return [int(n) if n.isdigit() else n for n in set(nums)]
        
        completion = completion.lower() 
        if "none" in completion and "all" not in completion:
            return ["None"]
        
        if "all" in completion and "none" not in completion:
            return ["All"]

    elif task == "SelectOne":
        choices = [str(i) for i in range(1, taskSize + 1)] + ["None", "All"]
        json_output = re.search("\{[^}]+}", completion)
        # first, try to get all the numbers inside the json_output
        if json_output:
            targets = re.findall(r"\b(" + "|".join(choices) + r")\b", json_output.group())
            
            if "All" in targets:
                if len(targets) == 1 or len(set(targets))-1 == taskSize:
                    targets = ["All"]
                else: # remove "All" that does not mean all the possible indices
                    targets = [t for t in targets if t != "All"]

            if "None" in targets:
                if len(set(targets)) != 1:
                    targets = [t for t in targets if t != "None"]
                else:
                    targets = ["None"]

            if len(targets) > 0:
                return [int(t) if t.isdigit() else t for t in set(targets)]
        else:
            # if the json_output is not found, try to get the numbers from the completion
            nums = re.findall(r"\d+", completion)
            if len(nums) > 0:
                return [int(n) if n.isdigit() else n for n in set(nums)]

            completion = completion.lower() 
            if "none" in completion and "all" not in completion:
                return ["None"]
            
            if "all" in completion and "none" not in completion:
                return ["All"]

    elif task == "SelectAll":  
        
        out = dict()
        try:
            json_output = re.search("\{[^}]+}", completion).group().replace("\'", "\"")
            json_output = re.sub(r"(?<=[^'\"])None", "\"None\"", json_output, flags=re.I)
            json_output = re.sub(r"(?<=[^'\"])All", "\"All\"", json_output, flags=re.I)
            json_output = json.loads(json_output)
            for k, v in json_output.items():
                if isinstance(v, list):
                    out[k.lower()] = v

                    if "All" in v and len(set(v)) - 1 == taskSize:
                        out[k.lower()] = ["All"]
                    else:
                        out[k.lower()] = [i for i in v if i != "All"]

                    if "None" in v and len(set(v)) == 1:
                        out[k.lower()] = ["None"]
                    else: 
                        out[k.lower()] = [i for i in v if i != "None"]
                else:
                    out[k.lower()] = [v]  
            return out

        except:
            # remove all newlines and quote marks
            completion = re.sub(r"\n", " ", completion)
            completion = re.sub(r"(\"|')", "", completion)
            
            # make each label: value pair stand in a new line
            for label in labels:
                completion = re.sub(rf"{label}( \S+)?:", f"\n{label}:", completion, flags=re.I)
            
            for label in labels:
                # extract the value for each label
                l = re.search(rf"{label}:.*?(\d|None|All).*", completion)
                if l:
                    targets = re.findall("(\d+|None|All)", l.group())
                    out[label.lower()] = [int(t) if t.isdigit() else t for t in set(targets)]
            if out:
                return out

            if model == "lmsys/vicuna-13b-v1.5" and benchmark == "MRPC":
                # allow one word after "non-paraphrases" and "paraphrases" and before ":"
                completion = re.sub(r"(non-paraphrases( \S+)?:|do not contain paraphrases:)", "\nNP:", completion, flags=re.I)
                completion = re.sub(r"paraphrases( \S+)?:", r"\nparaphrases:", completion, flags=re.I)

                if len(re.findall(r"paraphrases:.*?(\d|None|All)", completion)) == 1:
                    ps = re.search(r"paraphrases:.*?(\d|None|All).*", completion).group()
                    ps = re.findall("(\d+|None|All)", ps)
                    out["yes"] = [int(p) if p.isdigit() else p for p in set(ps)]

                if len(re.findall(r"NP:.*?(\d|None|All).*", completion)) == 1:
                    nps = re.search(r"NP:.*?(\d|None|All).*", completion).group()
                    nps = re.findall("(\d+|None|All)", nps, flags=re.I)
                    out["no"] = [int(p) if p.isdigit() else p for p in set(nps)]
                
                if out:
                    return out

    if print_unparsable:
        print(completion_copy)
    
    return "CANNOT_PARSE"


# helper function to inspect some sampled parsing results
def check_model_outputs(benchmarks, models, task="SingleClf", model_first=False, 
                        taskSize=None, sample_size=10, extract_func=None, 
                        SelectOne_json_output=True, res_dir="results/text classification",
                        show_answer=True, show_completion=True, print_unparsable=False, print_unparsable_only=False):
    if model_first:
        cols1, cols2 = models, benchmarks
        c1_name, c2_name = "Model", "Benchmark"
    else:
        cols1, cols2 = benchmarks, models
        c1_name, c2_name = "Benchmark", "Model"

    if extract_func is None:
        print_unparsable_only=False
    
    if print_unparsable_only:
        print_unparsable = True

    for col1 in cols1:
        print(f"{'='*50} {c1_name}: {col1} {'='*50}\n")
        for col2 in cols2:
            print(f"{'*'*50} {c2_name}: {col2} {'*'*50}\n")
            
            if model_first:
                benchmark, model = col2, col1
            else:
                benchmark, model = col1, col2 
            
            df = read_benchmark_results(benchmark, res_dir)

            if task in df.task.unique():
                df = df.copy()[df.task == task]

            if taskSize is not None and taskSize in df.taskSize.unique():
                df = df.copy()[df.taskSize == taskSize]
            
            for ix in df.sample(min(sample_size, len(df))).index:
                print("-" * 50 + f" Benchmark: {benchmark}; Row Index: {ix} " + "-" * 50)

                if task != "SingleClf":
                    print("taskSize ==>", df.at[ix, "taskSize"])

                targetLabel = df.at[ix, "targetLabel"]
                if "SelectOne" in task:
                    print("targetLabel ==>", targetLabel)

                completion = df.at[ix, f"{col2}-completion"] if not model_first else df.at[ix, f"{col1}-completion"]
                
                if show_completion and not print_unparsable_only:
                    print("Completion ==>", completion)
                    
                if extract_func is not None:
                    extracted = extract_func(completion, task, benchmark, model, df.at[ix, "taskSize"], 
                                             targetLabel, print_unparsable, SelectOne_json_output)
                    if not print_unparsable_only:
                        print("Extracted ==>", extracted)

                if show_answer and not print_unparsable_only:
                    print("Answer ==>", df.at[ix, "answer"])


def read_benchmark_results(benchmark, res_dir="results/text classification"):
    return pd.read_json(os.path.join(res_dir, f"{benchmark}.json"), lines=True)


def get_models(df):
    return [c.replace("-completion", "") for c in df.columns if "completion" in c]


# helper function to inspect **a** specific parsing result
def quick_check(benchmark, ix, task, model, return_completion=False, show_answer=True, 
                SelectOne_json_output=True, res_dir="results/text classification"):
    df = read_benchmark_results(benchmark, res_dir)
    task = df.at[ix, "task"]
    answer = df.at[ix, "answer"]
    taskSize = df.at[ix, "taskSize"]
    targetLabel = df.at[ix, "targetLabel"]
    comp = df.at[ix, f"{model}-completion"]
    extracted = extract_answer(comp, task, benchmark, model, taskSize=taskSize, 
                               targetLabel=targetLabel, print_unparsable=False, 
                               SelectOne_json_output=SelectOne_json_output)
    
    if return_completion:
        return comp

    print("Task size ==>", taskSize)
    print("Completion ==>", comp)
    print("Extracted ==>", extracted)

    if show_answer:
        print("Answer ==>", answer)


def parse_benchmark_model_completion(benchmark, models, tasks, CoT=[False, True], 
                                     SelectOne_json_output=True, res_dir="results/text classification"):
    df = read_benchmark_results(benchmark, res_dir)
    df = df.copy()[df.task.isin(tasks)]
    df = df.copy()[df.CoT.isin(CoT)]
    deployed_models = set(get_models(df))

    for model in models:
        assert model in deployed_models, f"{model} not deployed. No completions found."

    # df = df.drop(columns=[m for m in deployed_models if m not in models])

    out = []
    cols = ['benchmark', 'taskIndex', 'prompt', 'answer', 'targetLabel', 
            'task', '#shot', 'CoT', 'taskSize', "model", "completion", "parsed"]

    for ix in df.index:
        taskIndex = df.at[ix, "taskIndex"]
        prompt = df.at[ix, "prompt"]
        answer = df.at[ix, "answer"]
        targetLabel = df.at[ix, "targetLabel"]

        task = df.at[ix, "task"]
        num_shot = df.at[ix, "#shot"]
        cot = df.at[ix, "CoT"]
        taskSize = df.at[ix, "taskSize"]
        
        for model in models:
            completion = df.at[ix, f"{model}-completion"]
            parsed = extract_answer(completion, task, benchmark, model, taskSize, targetLabel, False, SelectOne_json_output)
            out.append([benchmark, taskIndex, prompt, answer, targetLabel, task, num_shot, 
                        cot, taskSize, model, completion, parsed])

    return pd.DataFrame(out, columns=cols)


def parse_benchmarks_models_completions(benchmarks, models, tasks, CoT=[False, True], 
                                        SelectOne_json_output=True, res_dir="results/text classification"):
    out = []
    for benchmark in benchmarks:
        out.append(parse_benchmark_model_completion(benchmark, models, tasks, CoT, SelectOne_json_output, res_dir))
    return pd.concat(out)
