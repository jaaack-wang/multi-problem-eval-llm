import numpy as np
import pandas as pd
from string import Template


def make_database(test_df, dev_df=None, num_instance=100, max_instance_size=100, benchmarkType="text classification"):
    '''Database contains questions sampled from benchmark datasets and are used for creating multi-question prompts. 
    
    Parameters:
        test_df: pd.DataFrame. Dataset from a benchmark used for composing multi-question prompts to probe LLMs.
        dev_df: pd.DataFrame. Dataset from a benchmark used to develop multi-question prompts or compose demonstrations.
        num_instance: int. Number of instances to compose multi-question prompts. Each instance contains multiple questions.
        max_instance_size: int. Maximum number of questions sampled from the benchmark dataset to compose an instance.
        benchmarkType: str. The type of benchmark dataset. Supported types are: ["text classification", "multi-choice", "free-response"].

    Returns:
        database: dict. A dictionary database object containing the data (text and label) from which the instances are constructed, 
                        the instances used for composing multi-question prompts, the prompt templates, and some meta-info.
    '''
    database = dict()
    database["num_instance"] = num_instance
    database["max_instance_size"] = max_instance_size

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
    '''Get indexed texts for multi-question prompts.'''
    indices = [prefix + str(i) + ". " for i in range(1, len(texts)+1)]
    texts += suffix
    return "\n".join(indices + texts)


def make_prompts_for_clf(database, task, dataType="test", propmtMode="zero-shot",
                         taskSize=1,  attr=None, label_attr_converter=None, num_instance=None):
    '''Make multi-question prompts based on text classification benchmarks.
    
    Parameters:
        - database: dict. A dictionary database object used for composing multi-question prompts.
        - task: str. The task to be performed. Supported tasks are: ["single_clf", "batch_clf", 
                     "index_selection_one_cat_a_time", "index_selection_all_cat_at_once"]
        - dataType: str. The type of data to be used from the **database** instances. Default to "test".
                        Note the testInstances in the database may not be a test set from the benchmark.
        - propmtMode: str. The prompt mode to be used. Default to "zero-shot". More modes to come.
        - taskSize: int. The number of questions to be included in a multi-question prompt. Default to 1, 
                         which is equivalent to a single-question prompt (i.e., single_clf).
        - attr: str. The attribute to be used for the "index_selection_one_cat_a_time" task. Default to None.
                     The attribute is highly related to the label of the classification task and should be indicated in the prompt template. 
        - label_attr_converter: function. A function to convert the label to the attribute value. Default to None, which equals str.lower.
        - num_instance: int. The number of instances to be used for composing multi-question prompts of size ::taskSize::. Default to 100.
    
    Returns:
        - pd.DataFrame. A DataFrame containing the multi-question prompts with answers as well as other important meta-info.
    '''

    max_taskSize = database["max_instance_size"]
    tasks = ["single_clf", "batch_clf", "index_selection_one_cat_a_time", "index_selection_all_cat_at_once", 
             "index_selection_one_cat_a_time_json", "index_selection_all_cat_at_once_adjusted"]

    assert task in tasks, f"Unrecognized task: {task}. Only support: {tasks}"
    assert 0 < taskSize <= max_taskSize, f"taskSize must be between 1 and {max_taskSize}"
    if task != "single_clf":
        assert taskSize > 1, "taskSize must be greater than 1 for all tasks except 'single_clf'"

    out = []
    cols = ["taskIndex", "prompt", "answer", "targetLabel", "task", "#shot", "CoT", "taskSize"]
    tmp = Template(database["promptTemplates"][propmtMode][task])
    data = database[f"{dataType}Data"]

    if label_attr_converter is None:
        label_attr_converter = str.lower

    if num_instance is None:
        num_instance = database["num_instance"] 
    elif isinstance(num_instance, int):
        num_instance = min(num_instance, database["num_instance"])
        
    if task == "single_clf":
        for ix, (text, label) in enumerate(zip(data["texts"], 
                                               data["labels"])):
            prompt = tmp.safe_substitute(text=text)
            out.append((ix+1, prompt, label, "NA", task, 0, "CoT" in propmtMode, 1))

    elif task == "batch_clf":
        for ix in range(1, num_instance + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            texts = data["texts"][instance][:taskSize]
            labels = data["labels"][instance][:taskSize]
            prompt = tmp.safe_substitute(num=taskSize, texts=get_indexed_texts(texts))
            out.append((ix, prompt, labels, "NA", task, 0, "CoT" in propmtMode, taskSize))
    
    elif task in ["index_selection_one_cat_a_time", "index_selection_one_cat_a_time_json"]:
        assert attr is not None, "attr must be provided for 'index_selection_one_cat_a_time' task"
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
                out.append((ix, prompt, answers, label, task, 0, "CoT" in propmtMode, taskSize))
    
    elif task in ["index_selection_all_cat_at_once", "index_selection_all_cat_at_once_adjusted"]:
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
            out.append((ix, prompt, answers, "NA", task, 0, "CoT" in propmtMode, taskSize))
            
    return pd.DataFrame(out, columns=cols)


def make_exemplars_for_clf(database, task, exemplar_templates, 
                           num_shot=2, taskSize=1, attr=None, label_attr_converter=None):

    max_taskSize = database["max_instance_size"]
    tasks = ["single_clf", "batch_clf", "index_selection_one_cat_a_time", "index_selection_all_cat_at_once"]

    assert task in tasks, f"Unrecognized task: {task}. Only support: {tasks}"
    assert 0 < taskSize <= max_taskSize, f"taskSize must be between 1 and {max_taskSize}"
    assert num_shot <= database["num_instance"], f"num_shot must be less than or equal to the number of instances in the database"

    if task != "single_clf":
        assert taskSize > 1, "taskSize must be greater than 1 for all tasks except 'single_clf'"

    out = []
    dataType = "dev"
    tmp = Template(exemplar_templates[task])
    data = database[f"{dataType}Data"]

    if label_attr_converter is None:
        label_attr_converter = str.lower
       
    if task == "single_clf":
        for ix, (text, label) in enumerate(zip(data["texts"][:num_shot], 
                                               data["labels"][:num_shot])):
            exemplar = tmp.safe_substitute(text=text, answer=label)
            out.append(exemplar)

    elif task == "batch_clf":
        for ix in range(1, num_shot + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            texts = data["texts"][instance][:taskSize]
            labels = data["labels"][instance][:taskSize]
            exemplar = tmp.safe_substitute(texts=get_indexed_texts(texts), answer="\n".join(labels))
            out.append(exemplar)
    
    elif task == "index_selection_one_cat_a_time":
        assert attr is not None, "attr must be provided for 'index_selection_one_cat_a_time' task"
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
                    "answer": "\n".join([str(a) for a in answers])}
            exemplar = tmp.safe_substitute(args)
            out.append(exemplar)
    
    elif task == "index_selection_all_cat_at_once":
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


def make_single_question_prompts(data, method="zero-shot"):
    '''Make single-question prompts along with their answers.
    
    Parameters:
        - data: dict. {
                        "dataset": "dataset anme",
                        "questions": [questions],
                        "answers": [answers]
                        "# samples": int,
                        "# exemplars": int,
                        "exemplars": {"questions": [questions], 
                                      "answers": [answers], 
                                      "rationales": [rationales]}
                        zero_shot_cot_trigger: trigger,
                        directAnswerTrigger: {"zero-shot": trigger, 
                                              "zero-shot-cot": trigger,
                                              "few-shot": trigger,
                                              "few-shot-cot": trigger}
                      }
        - method: str. Supported methods: ["zero-shot", "zero-shot-cot", "few-shot", "few-shot-cot"]
    
    Returns: list of single-question prompts along with their answers.
    Single-question prompt structure: 
        {few-shot-exemplars}

        Q: {question}
        A: {rationale} {direct answer trigger} (without answer)
    where each exemplar is a question-answer pair (with answer). rationale is for the two CoT methods.
    '''
    prompts = []
    answers = []

    exemplars = ""
        
    if "few-shot" in method:
        trigger = data["directAnswerTrigger"][method]

        for q, a, r in zip(data["exemplars"]["questions"], 
                        data["exemplars"]["answers"], 
                        data["exemplars"]["rationales"]):
            
            r = "" if "cot" not in method else " " + r
            exemplars += f"Q: {q}\nA:{r} {trigger} {a}\n\n"
        trigger = ""
    else:
        if method == "zero-shot-cot":
            trigger = " " + data["zero_shot_cot_trigger"]
        else:
            trigger = " " + data["directAnswerTrigger"][method]

    for q, a in zip(data["questions"], data["answers"]):
        prompt = exemplars + f"Q: {q}\nA:{trigger}"
        prompts.append(prompt)
        answers.append(a)
    
    return prompts, answers


def make_prompts_for_multi_choice(database, task, dataType="test", propmtMode="zero-shot",
                                  taskSize=1, num_instance=None):
    max_taskSize = database["max_instance_size"]
    tasks = ["single_question", "batch_questions", "index_selection_one_cat_a_time", "index_selection_all_cat_at_once"]

    assert task in tasks, f"Unrecognized task: {task}. Only support: {tasks}"
    assert 0 < taskSize <= max_taskSize, f"taskSize must be between 1 and {max_taskSize}"
    if task != "single_question":
        assert taskSize > 1, "taskSize must be greater than 1 for all tasks except 'single_question'"

    out = []
    cols = ["taskIndex", "prompt", "answer", "targetChoice", "task", "#shot", "CoT", "taskSize"]
    tmp = Template(database["promptTemplates"][propmtMode][task])
    data = database[f"{dataType}Data"]

    if num_instance is None:
        num_instance = database["num_instance"] 
    elif isinstance(num_instance, int):
        num_instance = min(num_instance, database["num_instance"])
        
    if task == "single_question":
        for ix, (question, answer) in enumerate(zip(data["questions"], 
                                                    data["answers"])):
            prompt = tmp.substitute(question=question)
            out.append((ix+1, prompt, answer, "NA", task, 0, "CoT" in propmtMode, 1))

    elif task == "batch_questions":
        for ix in range(1, num_instance + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            questions = data["questions"][instance][:taskSize]
            answers = data["answers"][instance][:taskSize]
            prompt = tmp.substitute(num=taskSize, questions=get_indexed_texts(questions, "Question ", "\n"))
            out.append((ix, prompt, answers, "NA", task, 0, "CoT" in propmtMode, taskSize))
    
    elif task == "index_selection_one_cat_a_time":

        for ix in range(1, num_instance + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            questions = data["questions"][instance][:taskSize]
            ans = data["answers"][instance][:taskSize]
            for choice in database["choices"]:
                prompt = tmp.substitute(num=taskSize, choice=choice, questions=get_indexed_texts(questions, "Question ", "\n"))
                
                answers = set(np.where(ans == choice)[0] + 1)
                if len(answers) == 0:
                    answers = {"None"}
                elif len(answers) == taskSize:
                    answers = {"All"}
                out.append((ix, prompt, answers, choice, task, 0, "CoT" in propmtMode, taskSize))
    
    elif task == "index_selection_all_cat_at_once":
        for ix in range(1, num_instance + 1):
            instance = database[f"{dataType}Instances"][f"#{ix}"]
            questions = data["questions"][instance][:taskSize]
            ans = data["answers"][instance][:taskSize]
            prompt = tmp.substitute({"num": taskSize, "questions": get_indexed_texts(questions, "Question ", "\n")})
            answers = dict()
            for choice in database["choices"]:
                answers[choice] = set(np.where(ans == choice)[0] + 1)
                if len(answers[choice]) == 0:
                    answers[choice] = {"None"}
                elif len(answers[choice]) == taskSize:
                    answers[choice] = {"All"}
            out.append((ix, prompt, answers, "NA", task, 0, "CoT" in propmtMode, taskSize))
            
    return pd.DataFrame(out, columns=cols)



def convert_reasoning_benchmark_to_database(data, num_instance=500, 
                                            max_instance_size=100, 
                                            duplicates_allowed=False):
    '''Convert the reasoning benchmark to the database format described above.'''
    database = data.copy()
    num_samples = data["# samples"]
    database["num_instance"] = num_instance
    database["max_instance_size"] = max_instance_size
    database["questions"] = np.array(data["questions"])
    database["answers"] = np.array(data["answers"])

    known = set()
    database[f"instances"] = dict()
    
    def get_sampled_indices():
        if duplicates_allowed:
            return np.random.randint(0, num_samples, max_instance_size)
        return np.random.permutation(num_samples)[:max_instance_size]

    for i in range(1, num_instance+1):
        while True:
            indices = get_sampled_indices()
            if tuple(indices) not in known:
                known.add(tuple(indices))
                database[f"instances"][f"#{i}"] = indices
                break
    
    return database


def numubering(texts, prefix):
    texts = [f"{prefix}{i+1}: {t}" for i, t in enumerate(texts)]
    return "\n\n".join(texts)


def make_to_multi_question_prompt_(questions, answers=None):
    qs = numubering(questions, "Q")
    multi_q_prompt = f"Questions\n\n{qs}\n\nAnswers"
    if answers is not None:
        multi_q_prompt += "\n\n" + numubering(answers, "A")
    
    return multi_q_prompt


def make_to_multi_question_prompt(questions, exemplars=None, method="zero-shot", 
                                  zero_shot_cot_trigger="Let's think step by step."):
    if method == "zero-shot":
        return make_to_multi_question_prompt_(questions)
    
    if method == "zero-shot-cot":
        return make_to_multi_question_prompt_(questions) + "\n\n" + zero_shot_cot_trigger
    
    if method in ["few-shot", "few-shot-cot"]:
        assert exemplars is not None, "Exemplars (questions-answers pairs) must be provided for few-shot prompts."
        
        out = []
        for qs, ans in exemplars:
            out.append(make_to_multi_question_prompt_(qs, ans))
        out.append(make_to_multi_question_prompt_(questions))
        return "\n\n".join(out)
    
    raise ValueError(f"Invalid method: {method}")


def make_to_multi_question_prompt_from_database(database, question_size, 
                                                method="zero-shot", num_instance=200):

    assert method in  ["zero-shot", "zero-shot-cot", "few-shot", "few-shot-cot"], \
        "Invalid method. Choose from 'zero-shot', 'zero-shot-cot', 'few-shot', 'few-shot-cot'."
    
    instances = database["instances"]
    questions = database["questions"]
    answers = database["answers"]
    exemplars = database["exemplars"]
    
    out = []
    cols = ["promptIdx", "prompt", "qsize", "answer"]
    zero_shot_cot_trigger = database["zero_shot_cot_trigger"]


    if "few-shot" in method:
        ex_qs = exemplars["questions"]

        if "cot" in method:
            trigger_fsc = database["directAnswerTrigger"]["few-shot-cot"]
            ex_ans_cot = [f"{r} {trigger_fsc} {a}" for r, a in zip(exemplars["rationales"], exemplars["answers"])]
            exs = [[ex_qs, ex_ans_cot]]
        else:
            trigger_fs = database["directAnswerTrigger"]["few-shot"]
            ex_ans = [f"{trigger_fs} {a}" for a in exemplars["answers"]]
            exs = [[ex_qs, ex_ans]]
    else:
        exs = None

    for i in range(1, num_instance+1):
        indices = instances[f"#{i}"]
        qs = questions[indices][:question_size]
        ans = answers[indices][:question_size]
        prompt = make_to_multi_question_prompt(qs, exs, method, zero_shot_cot_trigger)
        out.append([i, prompt, question_size, ans])
    
    return pd.DataFrame(out, columns=cols)