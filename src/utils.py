import re

import numpy as np
import pandas as pd


def KMPSearch(pat, txt):
    M = len(pat)
    N = len(txt)

    lps = [0] * M
    length = 0
    p = 1

    while p < M:
        if pat[p] == pat[length]:
            length += 1
            lps[p] = length
            p += 1
        elif length != 0:
                length = lps[length-1]
        else:
            lps[p] = 0
            p += 1

    i, j = 0, 0
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            return i - j

        elif i < N and pat[j] != txt[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return -1


def batchit(X, bs=1, droplast=False):
    batch = []
    for x in X:
        batch.append(x)
        if len(batch) == bs:
            yield batch
            batch.clear()
    if not droplast and len(batch) > 0:
        yield batch


def decode(text):
    for p1, p2 in [("/t", "\t"),
                    ("/n", "\n")]:
        text = text.replace(p1, p2)
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.replace('""', '"')


def encode(text):
    for p1, p2 in [("\t", "/t"),
                   ("\n", "/n")]:
        text = text.replace(p1, p2)
    return text


def compute_density(x, eps=7, p=4):
    xmax, xmin = x.max(), x.min()
    if not (xmax == 10.0 and xmin == 0.0):
        x = np.maximum(np.zeros_like(x), x)
        x = x / (x.max(axis=0, keepdims=True) + 1e-9)
        x = np.ceil(x * 10.)
    x = np.where(x <= eps, 0.0, x)
    l1 = x.sum(axis=-1)
    lp = (x ** p).sum(axis=-1) ** (1.0 / p) + 1e-9
    x = l1 / lp
    return x / (1e-9 + x.max())


def clean(tokens):
    new = []
    for token in tokens:
        for p in ["##", "\u0120", "\u2581"]:
            if len(token) > len(p) and token.startswith(p):
                token = token.replace(p, " ")
                break
        new.append(token)
    return new


def annotate(ids, prompt, text):
    text = decode(text)
    if text.startswith('"') and text.endswith('"'):
        text = text[i:-1]
    if len(text) == 0:
        return text
    cntx = text.split("&&&")[1] if "&&&" in text else ""
    prompts = []
    for prompt in prompt.split("\n"):
        prompts.extend(sent_tokenize(prompt))
    inst = set()
    for _ in text.split("&&&")[0].replace("\n", "|||").split("|||"):
        for __ in sent_tokenize(_):
            for p in prompts:
                if __ in p:
                    if p.startswith("Input:"):
                        p = p[6:].strip()
                    inst.add(p)
                    break
            else:
                raise RuntimeError("%s: %s ---> %s" % (ids, _, prompt))
    if "Output:" in inst:
        del inst["Output:"]
    return "|||".join(inst) + "&&&" + cntx

def sent_tokenize(text):
    return re.split(r'(?<=[.!?:])\s+', text)


def load_dataset(file, skip=0):
    ID, Prompt, Response, Instruct, Context, Type = [], [], [], [], [], []
    with open(file, encoding="utf8") as f:
        f.readline()
        for idx, row in enumerate(f):
            if len(row) == 1:
                continue
            if idx < skip:
                continue
            if row.count("\t") != 3:
                print("Error Line:" + row[:30])
            idx, p, r, i = row[:-1].split("\t")[:4]
            p = decode(p)
            ID.append(idx)
            Prompt.append(p)
            Response.append(decode(r))
            Instruct.append(annotate(idx, p, i))
    return ID, Prompt, Response, Instruct


def load_wiki_words(fpath, minfreq=0, only_ascii=False, only_alnum=False):
    words = {}
    with open(fpath, encoding="utf8") as f:
        for row in f:
            try:
                word, freq = row.strip().split()
                if only_ascii and not word.isascii():
                    continue
                if only_alnum and not word.isalnum():
                    continue
                for key in (str.lower, str.upper, str.capitalize):
                    temp = key(word)
                    if temp in words:
                        word = temp
                        break
                words[word] = words.get(word, 0.) + float(freq)
            except:
                pass
    return [_[0] for _ in words.items() if _[1] >= minfreq]
    print("Loading %d words from %s" % (len(words), fpath))


def merge_clean(expls):
    expls = "|||".join(expls)
    for c in string.ascii_lowercase:
        expls = expls.replace("\%s" % c, "/%s" % c)
    return expls


def tokens2text(tokenizer, tokens):
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokenizer.decode(ids, clean_up_tokenization_spaces=True)


def read_rawfile(file, columns=None, sep="\t"):
    with open(file, encoding="utf8") as f:
        if columns is None:
            column_row = f.readline()
            columns = column_row[:-1].split(sep)
        else:
            column_row = None
        values = [[] for _ in columns]
        for idx, row in enumerate(f):
            if row == column_row:
                continue
            if row.count(sep) != len(columns) - 1:
                print("Row=%d is failed!" % idx, row)
                continue
            for val, con in zip(row[:-1].split(sep), values):
                if val.isdigit():
                    val = float(val)
                con.append(val)
    return pd.DataFrame(dict(zip(columns, values)))
