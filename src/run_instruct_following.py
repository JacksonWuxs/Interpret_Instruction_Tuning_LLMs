import os
import tqdm
import time
import pickle
import argparse

import numpy as np

from generator import Generator
from utils import batchit, density, encode, clean, KMPSearch, load_dataset



def match_instruct_context_by_llama(prompt, instruct, tokens, tokenizer):
    masks = [0.] * len(tokens)
    for span in instruct:
        assert span in prompt, span + "|||" + prompt
        if prompt[max(0,  prompt.index(span) - 1)] == "\n":
            span = "\n" + span
        span = tokenizer.tokenize(span)
        while span[0] in {u"\u2581", "<0x0A>", u'\u2581"'}:
            span = span[1:]
        idx = KMPSearch(span, tokens)
        assert idx >= 0, str(span) + str(tokens)
        masks[idx:idx + len(span)] = [1.] * len(span)
    return np.array(masks)


def load_results(baseline_root, candidate_root, IDs, Instructs, Prompts, Responses):
    baselines = set(os.listdir(baseline_root))
    candidates = set(os.listdir(candidate_root))
    for file, instructs, prompt, resp in zip(IDs, Instructs, Prompts, Responses):
        if not file.endswith(".pkl"):
            file = file + ".pkl"
        if not (file in baselines and file in candidates):
            continue
        baseline = pickle.load(open(baseline_root + "/" + file, "rb"))
        candidate = pickle.load(open(candidate_root + "/" + file, "rb"))
        assert " ".join(baseline["Input"]) == " ".join(candidate["Input"])
        assert baseline["Sparsity"].shape == candidate["Sparsity"].shape
        yield file, instructs, prompt, resp, baseline, candidate


def collect(name, ids, prompts, references, batchsize=1, use_gold=False):
    model = Generator(name)
    bar = tqdm.tqdm(total=len(prompts) // batchsize)
    responses = []
    for batchID, batchP, batchR in zip(batchit(ids, batchsize),
                                       batchit(prompts, batchsize),
                                       batchit(references, batchsize)
                                       ):
        batchO = batchR.copy() if use_gold else model.generate(batchP) 
        batchI, batchO, batchE, _, _ = model.input_explain(batchP, batchO, b=0)
        
        for i in range(len(batchID)):
            #assert len(batchI[i]) == len(batchE[i])
            yield {"ID": batchID[i],
                   "Input": batchI[i],
                   "Reference": batchR[i],
                   "Output": batchO[i], 
                   "Explain": batchE[i],
                   }
        bar.update(1)
    bar.close()

def parsing_arguments():
    parser = argparse.ArgumentParser(prog="GIA")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parsing_arguments()
    print("Running dataset:", args.dataset)
    print("Saving to file: %s" % args.output)
    os.makedirs(args.output, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    ID, Prompt, Response = [], [], []
    for idx, prompt, response in zip(*load_dataset(args.dataset)[:3]):
        if not os.path.isfile(args.output + idx + '.pkl'):
            ID.append(idx)
            Prompt.append(prompt)
            Response.append(response)
    for item in collect(args.model, ID, Prompt, Response, use_gold=False):
        pickle.dump(item, open(args.output + item["ID"] + '.pkl', "wb"))
    
