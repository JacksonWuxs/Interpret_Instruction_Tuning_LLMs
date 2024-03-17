import sys
import collections

import pandas as pd
import numpy as np
import nltk
from scipy.stats import t as tdist
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

from utils import read_rawfile


class LemmaTokenizer:
    def __init__(self):
        self._lemmatize = nltk.wordnet.WordNetLemmatizer().lemmatize
        self._tokenize = nltk.word_tokenize
        self._tagging = nltk.pos_tag
        self._wordnet = nltk.corpus.wordnet
        self._stoppings = set(nltk.corpus.stopwords.words("english") +\
                              ['concept', 'related', 'context', 'different',
                               'specifically', 'similar', 'possibly', 'phrase'])
        self._suffix = {"ied": "y", "ies": "y", "ed": "", "s": "", "ing": ""}
        self._tagmaps = {"JJ": self._wordnet.ADJ, "RB": self._wordnet.ADV,
                         "VERB": self._wordnet.VERB, "VBZ": self._wordnet.VERB}
        
    def __call__(self, text):
        tokens = self._tokenize(text)
        lemmas = []
        for word, tag in self._tagging(tokens):
                tag = self._tagmaps.get(tag[0], self._wordnet.NOUN)
                word = self._lemmatize(word, tag)
                for x, y in self._suffix.items():
                    if word.endswith(x):
                        temp = word[:-len(x)] + y
                        if len(self._wordnet.synsets(temp)) > 1:
                            word = temp
                            break
                lemmas.append(word)
        return lemmas

    def clean(self, text):
        tokens = []
        for token in self(text.lower().replace('"', '')):
            if not token.isalnum():
                continue
            if token in self._stoppings:
                continue
            tokens.append(token)
        return tokens
            

def read(file, columns=None, maxrank=300, sep="\t"):
    data = read_rawfile(file, columns)
    #data = data[data.Type == "FFN" or data.Type == "down"]
    data = data[(data.Model == FT) | (data.Model == PT)]
    if maxrank is not None:
        data = data[data.Rank <= maxrank]

    data["Summary"] = data.Summary.apply(lambda x: x.lower().replace("word", "").replace("related to", "").rsplit(".")[0])
    data["Explainable"] = data.Summary.apply(lambda x: 0. if "cannot tell" in x.lower() else 1.)
    data["TopK"] = data.TopK.apply(lambda x: x.split("|||"))
    data["Group"] = data.Layer.apply(lambda x: x // 4)
    vicuna = data[data.Model == FT]
    llama = data[data.Model == PT]
    return data, vicuna, llama


def find_keywords(texts, tokenizer):
    concepts = collections.Counter()
    for src in texts:
        text = " ".join(tokenizer(src.lower()))
        tokens = []
        for t in tokenizer(text.lower()): 
            if t in tokenizer._stoppings:
                continue
            if len(t) <= 4:
                continue
            tokens.append(t)
        text = " ".join(tokens)
        for key in [("programming language",), ("computer science",), ("technology term",), ('starting', 'start'),
                    ("language proficiency",), ("software development",), ('foreign phrase', "foreign-language"),
                    ("different language", "foreign-language"), ("specific language", "foreign-language"),
                    ("foreign language",), ("language diversity", "foreign-language"), ('rhyme', "rhyming")]:
            if len(key) == 1:
                text = text.replace(key[0], key[0].replace(" ", "-"))
            else:
                text = text.replace(key[0], key[1])
        tokens = set()
        for token in sorted(text.split(), key=len, reverse=True):
            for check in tokens:
                if token in check:
                    break
            else:
                tokens.add(token)
        concepts.update(set(tokens))
    drop_list = []
    for k, v in concepts.items():
        if v <= 2:
            drop_list.append(k)
    for k in drop_list:
        del concepts[k]
    ttls = sum(concepts.values())
    for k, v in concepts.items():
        concepts[k] = v / ttls
    return concepts

tokenizer = LemmaTokenizer()
def common_substrings(strings):
    return " ".join([_ for _ in strings if "cannot tell" not in _.lower()]).lower()

def merging_seeds(seeds):
    assert len(seeds) > 2
    merge = seeds[0][["Words", "Model", "Size", "Layer", "Group", "Type", "Head", "Rank", "Score", "TopK"]].copy()
    merge["Explainable"] = np.where(np.mean([seed.Explainable for seed in seeds], axis=0) >= 0.5, 1.0, 0.0)
    merge["Summary"] = [common_substrings(strings) for strings in zip(*(seed.Summary for seed in seeds))]
    return merge


def linear_compare(name, categories, ploting=False):
    X = []
    for s in [0, 1, 2, 3, 4]:
        data, vicuna, llama = read(ROOT + r"%s_seed%d.tsv" % (name, s),
                                   ["Task", "Explainable", "Words", "Summary", "Model", "Size", "Layer", "Type", "Head", "Rank", "Score", "TopK"])
        x = []
        for task in categories:
            data[task] = data.Task.map(lambda x: 1.0 if task in x.lower() else 0.0)
            temp = data.groupby("Model")[task].mean()
            x.extend([temp[PT], temp[FT]])
        X.append(np.array(x).reshape(-1, 2))
    
    X = np.stack(X)
    for t, avg, std, X in zip(categories, X.mean(axis=0).tolist(), X.std(axis=0).tolist(), X.transpose(1, 0, 2)):
        print(t, "Mean:", avg, "Std:", std, "p-value:", ttest_ind(X[:, 0], X[:, 1]))

    if ploting:
        colors = ["blue", "green", "red", "purple"]
        marks = [("s", "--"), ("*", "-")]
        x = list(range(1, 1 + len(data.Group.unique())))
        plt.figure(figsize=(5, 4))
        for task, color in zip(categories, colors):
            for model, mark in zip([PT, FT], marks):
                temp = data[data.Model == model]
                vals = temp.groupby("Group")[task].mean()
                plt.plot(x, vals, color=color, marker=mark[0], linestyle=mark[1],
                         label=model.capitalize() + "-" + task.capitalize())
        plt.xlabel("Layers", fontsize=12)
        plt.ylabel('% of Concepts', fontsize=12)
        plt.xticks(x, ["1-4","5-8","9-12","13-16","17-20","21-24","25-28","29-32"], rotation=45)
        plt.legend(loc='center left', bbox_to_anchor=(0.1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(ploting)


def load_seeds(seeds=5):
    root = ROOT + "seed%d.tsv"
    datas, vicunas, llamas = [], [], []
    for s in range(seeds):
        data, vicuna, llama = read(root % (s,), ["Explainable", "Words", "Summary", "Model", "Size", "Layer", "Type", "Head", "Rank", "Score", "TopK"])
        datas.append(data)
        vicunas.append(vicuna)
        llamas.append(llama)
    vicunas = merging_seeds(vicunas)
    llamas = merging_seeds(llamas)
    return vicunas, llamas


def rank_change(X, Y, rank=2, reverse=False):
    # Convert dictionary to sorted list of (word, frequency) pairs
    sorted_x = sorted(X.items(), key=lambda x: x[1], reverse=True)
    sorted_y = sorted(Y.items(), key=lambda x: x[1], reverse=True)
    rank_x = {word: rank for rank, (word, _) in enumerate(sorted_x, 1)}
    rank_y = {word: rank for rank, (word, _) in enumerate(sorted_y, 1)}

    rank_changes = {word: -(rank_y[word] - rank_x.get(word, len(rank_x))) for word in (set(X) & set(Y))}
    sorted_rank_changes = sorted(rank_changes.items(), key=lambda x: x[1], reverse=reverse)
    return [word + '[%d]' % _ for word, _ in sorted_rank_changes[:rank]]


def freq_change(X, Y, rank=5, reverse=False):
    rank_x = {word: rank for rank, (word, _) in enumerate(sorted(X.items(), key=lambda _: _[1], reverse=True), 1)}
    rank_y = {word: rank for rank, (word, _) in enumerate(sorted(Y.items(), key=lambda _: _[1], reverse=True), 1)}

    rank_changes = {word: -(rank_y.get(word, len(rank_y)) - rank_x.get(word, len(rank_x))) for word in (set(X) & set(Y))}
    freq_changes = {word: Y.get(word, 0) - X.get(word, 0) for word in (set(X) & set(Y))}
    sorted_freq_changes = sorted(freq_changes.items(), key=lambda x: x[1], reverse=reverse)
    return [word + '[%d]' % rank_changes[word] for word, _ in sorted_freq_changes[:rank]]


def find_frequent_words(seeds=5):      
    vicunas, llamas = load_seeds(seeds)
    tokenizer = LemmaTokenizer()
    for group in vicunas.Group.unique():
        freqs = []
        for name, temp in [(FT, vicunas), (PT, llamas)]:
            temp = temp[temp.Group == group]
            temp = temp[temp.Explainable == 1]
            freqs.append(temp.Summary.tolist())
        vicuna = find_keywords(freqs[0], tokenizer)
        llama = find_keywords(freqs[1], tokenizer)
        print("\n\nGroup:", group)
        print("Rank Raising:", rank_change(llama, vicuna, rank=15, reverse=True))
        print("Rank Dropping:", rank_change(llama, vicuna, rank=15, reverse=False))
                    

if __name__ == "__main__":
    ROOT = sys.argv[1]
    FT, PT = "vicuna", "llama"
    print("Root:", ROOT)
    linear_compare("task", ['writing', 'math', 'coding', 'translation', 'none'])
    print("\n")
    linear_compare("concept", ['phonology', 'morphology', 'syntax', 'semantic'][::-1],
                    "ProportionOfLinguisticKnowledge.pdf")
    #find_frequent_words(5)

