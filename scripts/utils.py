import json
import pickle
import pandas as pd


def save_obj_as_pickle(obj, fp, print_msg=True):
    '''Save an object as a pickle file.'''
    with open(fp, "wb") as f:
        pickle.dump(obj, f)

        if print_msg:
            print(f"Saved object to {fp}")


def read_obj_from_pickle(fp, print_msg=True):
    '''Read an object from a pickle file.'''
    with open(fp, "rb") as f:
        obj = pickle.load(f)
        if print_msg:
            print(f"Read object from {fp}")
        return obj


def json_print(dic):
    '''Pretty print a dictionary as a JSON string.'''
    print(json.dumps(dic, indent=4))
    

def read_json(fp, print_msg=True):
    '''Read a JSON file and return the content as a dictionary.'''
    with open(fp, "rb") as f:
        out = json.load(f)

        if print_msg:
            print(fp + " has been loaded!")
    return out


def save_dict_as_json(dic, fp, indent=4, print_msg=True):
    '''Save a dictionary as a JSON file.'''
    with open(fp, "w") as f:
        json.dump(dic, f, indent=indent)

        if print_msg:
            print(fp + " saved!")