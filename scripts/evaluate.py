import re
import pandas as pd

import sys
import pathlib
# import from local script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from parsing import parse_benchmark_model_completion


def calculate_jaccardSim(set1, set2):
    if set1 == set2: # so in case both are empty sets, jaccard sim = 1.0
        return 1.0
    return len(set(set1) & set(set2)) / len(set(set1) | set(set2))


def evaluate_parsed_benchmark_model_performance(df):
    out = []
    cols = ['benchmark', 'taskIndex', 'prompt', 'answer', 'targetLabel', 
            'task', '#shot', 'CoT', 'taskSize', "model", "completion", "parsed", 
            "perTaskAccu", "jaccardSim", "#dif", "#contradictions", "#nonExcludedMiddles"]
    
    for task in df.task.unique():
        sub = df.copy()[df.task == task]
        if task != "SelectOne":
            for ix in sub.index:
                benchmark = df.at[ix, "benchmark"]
                taskIndex = df.at[ix, "taskIndex"]
                prompt = df.at[ix, "prompt"]
                answer = df.at[ix, "answer"]
                targetLabel = df.at[ix, "targetLabel"]

                task = df.at[ix, "task"]
                num_shot = df.at[ix, "#shot"]
                cot = df.at[ix, "CoT"]
                taskSize = df.at[ix, "taskSize"]
                model = df.at[ix, "model"]
                completion = df.at[ix, "completion"]
                parsed = df.at[ix, "parsed"]

                jaccardSim = None
                num_dif = None
                num_contradictions = None
                num_non_excluded_middles = None

                if task == "SingleClf":
                    perTaskAccu = float(parsed.lower() == answer.lower())

                elif task == "BatchClf":
                    n, m = len(answer), len(parsed)
                    perTaskAccu = sum([float(a == p) for a, p in zip(answer, parsed)]) / n
                    num_dif = n - m
                
                elif task == "SelectAll":

                    answer_ix2cat = dict()
                    parsed_ix2cat = dict()
                    jaccardSim = dict()
                    num_contradictions = 0
                    num_non_excluded_middles = 0

                    for cat, ans_values in answer.items():
                        if ans_values == ["All"]:
                            ans_values = set(range(1, taskSize+1))
                        elif ans_values == ["None"]:
                            ans_values = set()
                        for v in ans_values:
                            answer_ix2cat[v] = [cat]
                        
                        # parsed may be equal to "CANNOT_PARSE"
                        pred_values = set(parsed.get(cat, [])) if isinstance(parsed, dict) else set() 
                        if pred_values == {"All"}:
                            pred_values = set(range(1, taskSize+1))
                        elif pred_values == {"None"}:
                            pred_values = set()
                        for v in pred_values:
                            parsed_ix2cat[v] = parsed_ix2cat.get(v, []) + [cat]
                        
                        jaccardSim[cat] = calculate_jaccardSim(ans_values, pred_values)
                    
                    ex_match = 0
                    for ix, ans_cat in answer_ix2cat.items():
                        pred_cat = parsed_ix2cat.get(ix, [])
                        if ans_cat == pred_cat:
                            ex_match += 1
                        elif len(pred_cat) == 0:
                            num_non_excluded_middles += 1
                        else:
                            num_contradictions += (len(pred_cat) - 1)

                    perTaskAccu = ex_match / taskSize
                
                out.append([benchmark, taskIndex, prompt, answer, targetLabel, task, num_shot, cot, taskSize, model, completion, parsed,
                            perTaskAccu, jaccardSim, num_dif, num_contradictions, num_non_excluded_middles])
            
        else:
            # this is because the task can be SelectOne
            for benchmark in sub.benchmark.unique():
                subsub = sub.copy()[sub.benchmark == benchmark]
                # sorted the dataframe to make sure the same taskIndex from the same conditions appear in consecutive rows 
                # this also greatly reduces the time complexity of the code by avoiding many nested loops
                subsub_sorted = subsub.sort_values(["benchmark", "task", "taskSize", "#shot", "CoT", "model", "taskIndex"])
                stride = subsub_sorted["targetLabel"].unique().size

                for l in range(0, len(subsub_sorted), stride):
                    answer_ix2cat = dict()
                    parsed_ix2cat = dict()
                    jaccardSims = []

                    for ix in subsub_sorted.index[l:l+stride]: # belong to the same task index
                        cat = subsub_sorted.at[ix, "targetLabel"]
                        answer = set(subsub_sorted.at[ix, "answer"])
                        parsed = set(subsub_sorted.at[ix, "parsed"])
                        taskSize = subsub_sorted.at[ix, "taskSize"]

                        if answer == {"None"}:
                            answer = set()
                        elif answer == {"All"}:
                            answer = set(range(1, taskSize+1))

                        if parsed == {"None"}:
                            parsed = set()
                        elif parsed == {"All"}:
                            parsed = set(range(1, taskSize+1))
                        
                        jaccardSims.append(calculate_jaccardSim(answer, parsed))
                        for a in answer:
                            answer_ix2cat[a] = [cat]
                        
                        for p in parsed:
                            # just in case
                            if not isinstance(p, int) and p.isdigit():
                                p = int(p)
                            parsed_ix2cat[p] = parsed_ix2cat.get(p, []) + [cat]
                    
                    ex_match = 0
                    num_dif = None
                    num_contradictions = 0
                    num_non_excluded_middles = 0
                    task = subsub_sorted.at[ix, "task"]
                    taskIndex = subsub_sorted.at[ix, "taskIndex"]
                    num_shot = subsub_sorted.at[ix, "#shot"]
                    cot = subsub_sorted.at[ix, "CoT"]
                    model = subsub_sorted.at[ix, "model"]

                    for ix, ans_cat in answer_ix2cat.items():
                        pred_cat = parsed_ix2cat.get(ix, [])
                        if ans_cat == pred_cat:
                            ex_match += 1
                        elif len(pred_cat) == 0:
                            num_non_excluded_middles += 1
                        else:
                            num_contradictions += (len(pred_cat) - 1)

                    perTaskAccu = ex_match / taskSize

                    for j, ix in enumerate(subsub_sorted.index[l:l+stride]): 
                        cat = subsub_sorted.at[ix, "targetLabel"]
                        answer = subsub_sorted.at[ix, "answer"]
                        parsed = subsub_sorted.at[ix, "parsed"]
                        prompt = subsub_sorted.at[ix, "prompt"]
                        completion = subsub_sorted.at[ix, "completion"]
                        out.append([benchmark, taskIndex, prompt, answer, cat, task, num_shot, cot, taskSize, model, completion, parsed,
                                    perTaskAccu, jaccardSims[j], num_dif, num_contradictions, num_non_excluded_middles])

    return pd.DataFrame(out, columns=cols)


def get_parse_rate_and_performance(benchmarks, models, tasks=["SingleClf"], CoT=[False, True], 
                                   return_evaluated_df=False, SelectOne_json_output=True,  
                                   res_dir="results/text classification"):
    out = []
    cols = ["benchmark", "task", "taskSize", "model", "CoT", "#shot", "parse_rate", "performance"]
    
    if return_evaluated_df:
        eval_dfs = []

    for benchmark in benchmarks:
        df = parse_benchmark_model_completion(benchmark, models, tasks, CoT, SelectOne_json_output, res_dir)
        df = evaluate_parsed_benchmark_model_performance(df)
        if return_evaluated_df:
            eval_dfs.append(df)

        for task in tasks:
            sub = df.copy()[df.task == task]
            taskSizes = sub.taskSize.unique()

            for taskSize in taskSizes:

                subsub = sub.copy()[sub.taskSize == taskSize]

                for cot in CoT:
                    subsubsub = subsub.copy()[subsub.CoT == cot]
                    for num_shot in subsubsub["#shot"].unique():
                        subsubsubsub = subsubsub.copy()[subsubsub["#shot"] == num_shot]
                        for model in models:
                            parsed = subsubsubsub[subsubsubsub.model == model]["parsed"]
                            parse_rate = (parsed != "CANNOT_PARSE").mean()
                            performance = subsubsubsub[subsubsubsub.model == model]["perTaskAccu"].mean()
                            out.append([benchmark, task, taskSize, model, cot, num_shot, parse_rate, performance])
    
    out = pd.DataFrame(out, columns=cols)

    if return_evaluated_df:
        eval_dfs = pd.concat(eval_dfs).reset_index(drop=True)
        return out, eval_dfs
    
    return out
