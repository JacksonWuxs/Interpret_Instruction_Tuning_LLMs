import sys
import collections

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from wordfreq import word_frequency
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from utils import read_rawfile

        
def sampling_verbs(n=30, excludes=None):
    if excludes is None:
        excludes = []
    excludes = set(excludes) | set(stopwords.words("english"))
    fullist = set()
    for synset in wn.all_synsets(pos=wn.VERB):
        if len(synset.examples()) == 0:
            continue
        word = synset.lemmas()[0].name().lower()
        if word not in excludes:
            freq = word_frequency(word, 'en')
            fullist.add((word, freq))
    fullist = sorted(fullist, key=lambda x: x[1], reverse=True)
    return [_[0] for _ in fullist[:n]]
    


if __name__ == "__main__":
    pretrain, finetune = "llama", "vicuna"
    data = read_rawfile(sys.argv[1])
    data["TopK"] = data["TopK"].apply(lambda x: ["=".join(sorted(_.split("="))).lower() for _ in x.split("|||")])
    data = data[["Model", "Layer", "Head", "Dim", "TopK"]]
    
    vicuna = data[data.Model == finetune].rename(columns={"TopK": "Vicuna"})
    llama = data[data.Model == pretrain].rename(columns={"TopK": "Llama"})
    del vicuna["Model"], llama["Model"]
    data = pd.merge(llama, vicuna, on=["Layer", "Head", "Dim"])
    data = data[["Layer", "Head", "Dim", "Vicuna", "Llama"]]
    
    vicuna_head_knowledge = data.groupby(["Layer", "Head"]).apply(lambda x: [_[0] for _ in collections.Counter(__ for _ in x.Vicuna for __ in _).most_common() if _[1] > 1])
    llama_head_knowledge = data.groupby(["Layer", "Head"]).apply(lambda x: [_[0] for _ in collections.Counter(__ for _ in x.Llama for __ in _).most_common() if _[1] > 1])

    knowledge = pd.DataFrame({"Vicuna": vicuna_head_knowledge, "Llama": llama_head_knowledge}).reset_index()
    knowledge["Group"] = knowledge.Layer.apply(lambda x: x // 8)
    
    instruct1 = [ 
                "translate", "explain", "summarize", "retrieve",
                "revise", 'generate', 'describe', 'classify', 'create', 
                "evaluate", "correct", "develop", 
                "identify", "analyze", "compose", "demonstrate", "interpret", 
                "design", "solve", "follow", "clarify", "say", "help", "act",
                "recommend", "estimate", "edit", "format", "repeat"
                ]
    instruct2 = ["write", "give", "find", "create", "make", "describe", "design", "generate", "classify", "have",
                 "explain", "tell", "identify", "output", "predict", "detect",                 ]
    instruct = set(instruct1 + instruct2)
    general = sampling_verbs(3000, instruct)


    knowledge['Finetune_Word'] = knowledge.Vicuna.apply(lambda x: collections.Counter(__ for _ in x for __ in _.split("=")))
    knowledge['Pretrain_Word'] = knowledge.Llama.apply(lambda x: collections.Counter(__ for _ in x for __ in _.split("=")))

    for group in knowledge.Group.unique():
        temp = knowledge[knowledge.Group == group]
        diffs = []
        for wordlist in [instruct, general]:
            diff = []
            for word in wordlist:
                vicuna = temp.Finetune_Word.apply(lambda x: x[word]).tolist()
                llama = temp.Pretrain_Word.apply(lambda x: x[word]).tolist()
                pairs = []
                for v, l in zip(vicuna, llama):
                    if v > l:
                        pairs.append(1)
                    elif v == l:
                        # we ignore the heads having no change respect to a target word
                        continue
                    else:
                        pairs.append(0)
                if len(pairs) > 0:
                    diff.append(np.mean(pairs))
            diffs.append(diff)
        print("Instruct - Group=%s | Mean=%.6f | Std=%.6f | Size=%d" % (group, np.mean(diffs[0]), np.std(diffs[0]), len(diffs[0])))
        print("General - Group=%s | Mean=%.6f | Std=%.6f | Size=%d" % (group, np.mean(diffs[1]), np.std(diffs[1]), len(diffs[1])))
        print(ttest_ind(diffs[0], diffs[1], equal_var=False))
        print("\n")
