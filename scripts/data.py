import numpy as np
import pandas as pd
from string import Template


def make_database(test_df, dev_df=None, num_instance=100, max_instance_size=100, benchmarkType="text classification"):
    '''Database contains problems sampled from benchmark datasets and are used for creating multi-problem prompts. 
    
    Parameters:
        test_df: pd.DataFrame. Dataset from a benchmark used for composing multi-problem prompts to probe LLMs.
        dev_df: pd.DataFrame. Dataset from a benchmark used to develop multi-problem prompts or compose demonstrations.
        num_instance: int. Number of instances to compose multi-problem prompts. Each instance contains multiple problems.
        max_instance_size: int. Maximum number of problems sampled from the benchmark dataset to compose an instance.
        benchmarkType: str. The type of benchmark dataset. Supported types are: ["text classification", "multi-choice", "free-response"].

    Returns:
        database: dict. A dictionary database object containing the data (text and label) from which the instances are constructed, 
                        the instances used for composing multi-problem prompts, the prompt templates, and some meta-info.
    '''
    database = dict()
    database["num_instance"] = num_instance
    database["max_instance_size"] = max_instance_size
    assert max_instance_size < len(test_df), f"max_instance_size must be less than the number of instances in the dataset for variations"

    if benchmarkType == "text classification":
        database["labels"] = sorted(test_df.label.unique().tolist())
        text_col, answer_col = "text", "label"
    elif benchmarkType == "multi-choice":
        database["choices"] = sorted(test_df.answer.unique().tolist())
        text_col, answer_col = "question", "answer"
    elif benchmarkType == "free-response":
        text_col, answer_col = "question", "answer"

    def make(name, df):
        database[f"{name}Data"] = dict()
        database[f"{name}Data"][f"{text_col}s"] = df[text_col].to_numpy()
        database[f"{name}Data"][f"{answer_col}s"] = df[answer_col].to_numpy()

        if "rationale" in df.columns:
            database[f"{name}Data"]["rationales"] = df["rationale"].to_numpy()
        
        known = set()
        database[f"{name}Instances"] = dict()
        for i in range(1, num_instance+1):
            
            while True:
                sample = df.sample(max_instance_size)
                indices = sample.index.to_numpy()
                if tuple(indices) not in known:
                    known.add(tuple(indices))
                    database[f"{name}Instances"][f"#{i}"] = indices
                    break

    make("test", test_df)
    if dev_df is not None:
        make("dev", dev_df)
    
    return database


def get_indexed_texts(texts: np.ndarray, prefix="", suffix=""):
    '''Get indexed texts for multi-problem prompts.'''
    indices = [prefix + str(i) + ". " for i in range(1, len(texts)+1)]
    texts += suffix
    return "\n".join(indices + texts)


def make_prompts_for_clf(database, task, dataType="test", propmtMode="zero-shot",
                         taskSize=1,  attr=None, label_attr_converter=None, num_instance=None):
    '''Make multi-problem prompts based on text classification benchmarks.
    
    Parameters:
        - database: dict. A dictionary database object used for composing multi-problem prompts.
        - task: str. The task to be performed. Supported tasks are: ["SingleClf", "BatchClf", 
                     "SelectOne", "SelectAll"]
        - dataType: str. The type of data to be used from the **database** instances. Default to "test".
                        Note the testInstances in the database may not be a test set from the benchmark.
        - propmtMode: str. The prompt mode to be used. Default to "zero-shot". More modes to come.
        - taskSize: int. The number of problems to be included in a multi-problem prompt. Default to 1, 
                         which is equivalent to a single-problem prompt (i.e., SingleClf).
        - attr: str. The attribute to be used for the "SelectOne" task. Default to None.
                     The attribute is highly related to the label of the classification task and should be indicated in the prompt template. 
        - label_attr_converter: function. A function to convert the label to the attribute value. Default to None, which equals str.lower.
        - num_instance: int. The number of instances to be used for composing multi-problem prompts of size ::taskSize::. Default to 100.
    
    Returns:
        - pd.DataFrame. A DataFrame containing the multi-problem prompts with answers as well as other important meta-info.
    '''

    max_taskSize = database["max_instance_size"]
    tasks = ["SingleClf", "BatchClf", "SelectOne", "SelectAll"]

    assert task in tasks, f"Unrecognized task: {task}. Only support: {tasks}"
    assert 0 < taskSize <= max_taskSize, f"taskSize must be between 1 and {max_taskSize}"
    if task != "SingleClf":
        assert taskSize > 1, "taskSize must be greater than 1 for all tasks except 'SingleClf'"

    out = []
    cols = ["taskIndex", "prompt", "answer", "targetLabel", "task", "#shot", "CoT", "taskSize"]
    tmp = Template(database["promptTemplates"][propmtMode][task])
    num_shot = int(propmtMode.split("-")[0])
    data = database[f"{dataType}Data"]

    if label_attr_converter is None:
        label_attr_converter = str.lower

    if num_instance is None:
        num_instance = database["num_instance"] 
    elif isinstance(num_instance, int):
        num_instance = min(num_instance, database["num_instance"])
        
    if task == "SingleClf":
        for ix, (text, label) in enumerate(zip(data["texts"], 
                                               data["labels"])):
            prompt = tmp.safe_substitute(text=text)
            out.append((ix+1, prompt, label, "NA", task, num_shot, "CoT" in propmtMode, 1))

    elif task == "BatchClf":
        for ix in range(1, num_instance + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            texts = data["texts"][instance][:taskSize]
            labels = data["labels"][instance][:taskSize]
            prompt = tmp.safe_substitute(num=taskSize, texts=get_indexed_texts(texts))
            out.append((ix, prompt, labels, "NA", task, num_shot, "CoT" in propmtMode, taskSize))
    
    elif task in ["SelectOne", "SelectOne"]:
        assert attr is not None, "attr must be provided for 'SelectOne' task"
        assert f"${attr}" in tmp.template, f"attr: {attr} not found in the template"

        for ix in range(1, num_instance + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            texts = data["texts"][instance][:taskSize]
            labels = data["labels"][instance][:taskSize]
            for label in database["labels"]:
                args = {"num": taskSize, "texts": get_indexed_texts(texts), attr: label_attr_converter(label)}
                prompt = tmp.safe_substitute(args)
                
                answers = set(np.where(labels == label)[0] + 1)
                if len(answers) == 0:
                    answers = {"None"}
                elif len(answers) == taskSize:
                    answers = {"All"}
                out.append((ix, prompt, answers, label, task, num_shot, "CoT" in propmtMode, taskSize))
    
    elif task == "SelectAll":
        for ix in range(1, num_instance + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            texts = data["texts"][instance][:taskSize]
            labels = data["labels"][instance][:taskSize]
            prompt = tmp.safe_substitute({"num": taskSize, "texts": get_indexed_texts(texts)})
            answers = dict()
            for label in database["labels"]:
                l = label.lower()
                answers[l] = set(np.where(labels == label)[0] + 1)
                if len(answers[l]) == 0:
                    answers[l] = {"None"}
                elif len(answers[l]) == taskSize:
                    answers[l] = {"All"}
            out.append((ix, prompt, answers, "NA", task, num_shot, "CoT" in propmtMode, taskSize))
            
    return pd.DataFrame(out, columns=cols)


def make_exemplars_for_clf(database, task, exemplar_templates, 
                           num_shot=2, taskSize=1, attr=None, label_attr_converter=None):

    max_taskSize = database["max_instance_size"]
    tasks = ["SingleClf", "BatchClf", "SelectOne", "SelectAll"]

    assert task in tasks, f"Unrecognized task: {task}. Only support: {tasks}"
    assert 0 < taskSize <= max_taskSize, f"taskSize must be between 1 and {max_taskSize}"
    assert num_shot <= database["num_instance"], f"num_shot must be less than or equal to the number of instances in the database"

    if task != "SingleClf":
        assert taskSize > 1, "taskSize must be greater than 1 for all tasks except 'SingleClf'"

    out = []
    dataType = "dev"
    tmp = Template(exemplar_templates[task])
    data = database[f"{dataType}Data"]

    if label_attr_converter is None:
        label_attr_converter = str.lower
       
    if task == "SingleClf":
        for ix, (text, label) in enumerate(zip(data["texts"][:num_shot], 
                                               data["labels"][:num_shot])):
            exemplar = tmp.safe_substitute(text=text, answer=label)
            out.append(exemplar)

    elif task == "BatchClf":
        for ix in range(1, num_shot + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            texts = data["texts"][instance][:taskSize]
            labels = data["labels"][instance][:taskSize]
            exemplar = tmp.safe_substitute(texts=get_indexed_texts(texts), answer="\n".join(labels))
            out.append(exemplar)
    
    elif task == "SelectOne":
        assert attr is not None, "attr must be provided for 'SelectOne' task"
        assert f"${attr}" in tmp.template, f"attr: {attr} not found in the template"

        for ix in range(1, num_shot + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            texts = data["texts"][instance][:taskSize]
            labels = data["labels"][instance][:taskSize]
            label = database["labels"][ix % num_shot]
            answers = np.where(labels == label)[0] + 1
            if len(answers) == 0:
                answers = ["None"]
            elif len(answers) == taskSize:
                answers = ["All"]
            
            args = {"texts": get_indexed_texts(texts), attr: label_attr_converter(label), 
                    "answer": [a for a in answers]}
            exemplar = tmp.safe_substitute(args)
            out.append(exemplar)
    
    elif task == "SelectAll":
        for ix in range(1, num_shot + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            texts = data["texts"][instance][:taskSize]
            labels = data["labels"][instance][:taskSize]
            answers = dict()
            for label in database["labels"]:
                l = label.lower()
                answers[l] = list(np.where(labels == label)[0] + 1)
                if len(answers[l]) == 0:
                    answers[l] = ["None"]
                elif len(answers[l]) == taskSize:
                    answers[l] = ["All"]
            exemplar = tmp.safe_substitute({"texts": get_indexed_texts(texts), "answer": str(answers)})
            out.append(exemplar)
            
    return "\n\n".join(out)
