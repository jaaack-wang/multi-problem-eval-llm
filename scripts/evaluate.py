import re
import pandas as pd

try:
    from parsing import parse_benchmark_model_completion
except:
    from scripts.parsing import parse_benchmark_model_completion


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
        if not task.startswith("index_selection_one_cat_a_time"):
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

                if task == "single_clf":
                    perTaskAccu = float(parsed.lower() == answer.lower())

                elif task == "batch_clf":
                    n, m = len(answer), len(parsed)
                    perTaskAccu = sum([float(a == p) for a, p in zip(answer, parsed)]) / n
                    num_dif = n - m
                
                elif task.startswith("index_selection_all_cat_at_once"):

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
            # this is because the task can be: index_selection_one_cat_a_time or index_selection_one_cat_a_time_json
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
                            parsed = set(range(1, taskSize))
                        
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


def get_max_random_baseline(df):
    sub = df.copy()[df.task == "single_clf"].drop_duplicates(subset=["prompt", "answer"])
    return max(sub["answer"].value_counts()) / len(sub)


# def get_parse_rate_and_performance(benchmarks, models, tasks=["single_clf"], return_evaluated_df=False):
#     out = []
#     cols = ["benchmark", "task", "taskSize", "model", "parse_rate", "performance"]
    
#     if return_evaluated_df:
#         eval_dfs = []

#     for benchmark in benchmarks:
#         df = parse_benchmark_model_completion(benchmark, models, tasks)
#         if "single_clf" in tasks:
#             out.append([benchmark, "single_clf", 1, "random baseline", "-", get_max_random_baseline(df)])

#         df = evaluate_parsed_benchmark_model_performance(df)
#         if return_evaluated_df:
#             eval_dfs.append(df)

#         for task in tasks:
#             sub = df.copy()[df.task == task]
#             taskSizes = sub.taskSize.unique()

#             for taskSize in taskSizes:
#                 subsub = sub.copy()[sub.taskSize == taskSize]
#                 for model in models:
#                     parsed = subsub[subsub.model == model]["parsed"]
#                     parse_rate = (parsed != "CANNOT_PARSE").mean()
#                     accu = subsub[subsub.model == model]["perTaskAccu"].mean()
#                     out.append([benchmark, task, taskSize, model, parse_rate, accu])

#     out = pd.DataFrame(out, columns=cols)

#     if return_evaluated_df:
#         eval_dfs = pd.concat(eval_dfs).reset_index(drop=True)
#         return out, eval_dfs
    
#     return out

def get_parse_rate_and_performance(benchmarks, models, tasks=["single_clf"], CoT=[False, True], return_evaluated_df=False):
    out = []
    cols = ["benchmark", "task", "taskSize", "model", "CoT", "#shot", "parse_rate", "performance"]
    
    if return_evaluated_df:
        eval_dfs = []

    for benchmark in benchmarks:
        df = parse_benchmark_model_completion(benchmark, models, tasks)
        if "single_clf" in tasks:
            out.append([benchmark, "single_clf", 1, "random baseline", "-", get_max_random_baseline(df)])

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


def answer_cleansing(pred, method, dataset, direct_answer_trigger_for_fewshot):
    '''Post process LLM completions for the evaluation based on single-question prompts. 
        Adapted from https://github.com/kojima-takeshi188/zero_shot_cot'''
    
    if len(pred) != 0 and pred[-1] in ["A", "B", "C", "D", "E", "F"]:
        pred += "."
    if method in ("few-shot", "few-shot-cot"):
        preds = pred.split(direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]

    # add "(?=\W)" to avoid extracting answer options from a capitalized names for all multiple choice datasets
    if dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'(A|B|C|D|E)(?=\W)', pred)
    elif dataset == "bigbench_date":
        pred = re.findall(r'(A|B|C|D|E|F)(?=\W)', pred)
    elif dataset in ("object_tracking"):
        # pred = re.sub(r"Alice|Bob|Claire", "", pred)
        pred = re.findall(r'(A|B|C)(?=\W)', pred)
    elif dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if method in ("few-shot", "few-shot-cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif method in ("zero-shot", "zero-shot-cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        
    return pred