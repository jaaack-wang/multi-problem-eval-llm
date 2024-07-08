import os
import tiktoken
from openai import OpenAI
from time import sleep


model = "gpt-3.5-turbo-0125"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
GPTbase = {"babbage-002", "davinci-002"}


def get_num_of_tokens(text, model=model):
    '''Get token number estimation for a given LLM.'''
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def get_completion(prompt, model=model,
                   max_tokens=None,
                   temperature=0.0,
                   return_all=False,
                   api_key=None):
    '''Get prompt completion given a LLM model via Openai API and TogetherAI API. 
    
    Supported models:
        - From OpenAI, see: https://platform.openai.com/docs/models
        - From Together AI, see: https://docs.together.ai/docs/inference-models

    Args:
        - prompt: str. The prompt to be completed
        - model: str. The model to be used
        - max_tokens: int. The maximum number of output tokens to generate. Useful 
                        for controlling the length of the completion to avoid overcharge.
        - temperature: float. The value to control the randomness of the completion. Default to 0.0.
        - return_all: bool. If True, return the full response object. Default to False (only output string).
    '''
    if "gpt" in model or model in GPTbase: # max_retries defaults to 2
        client = OpenAI(api_key=OPENAI_API_KEY if api_key is None else api_key, max_retries=2)
    else:
        client = OpenAI(api_key=TOGETHER_API_KEY if api_key is None else api_key, max_retries=2,
                        base_url='https://api.together.xyz/v1')
    while True:
        
        try:
            if model in GPTbase:
                response = client.completions.create(
                    model=model, max_tokens=max_tokens,
                    temperature=temperature, top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    prompt=prompt)
            else:
                response = client.chat.completions.create(
                    model=model, max_tokens=max_tokens,
                    temperature=temperature, top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    messages=[{"role": "user", "content": prompt}])
            break
        except Exception as e:
            print("Running into problem:", e)
            print("Retrying...")
            sleep(5)
    
    if return_all:
        return response

    if model in GPTbase:
        return response.choices[0].text
        
    return response.choices[0].message.content
