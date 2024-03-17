import time
import re
import string
import json
import sys
import concurrent
import multiprocessing

import tqdm
import openai
import filelock


STORED_FILE = None #"./cache_myself.txt"


def synchronize(func, iters, batch_size=None, workers=None):
    if workers is None:
        workers = multiprocessing.cpu_count() * 2
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for batch in batchit(iters, batch_size):
            yield pool.map(func, batch)


def batchit(corpus, size=128):
    assert hasattr(corpus, "__iter__")
    assert size is None or isinstance(size, int) and size > 0
    batch = []
    for row in corpus:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch.clear()
    if len(batch) > 0:
        yield batch



class _APISetup:
    def __init__(self, secret_key, engine, function, max_retry=None, cool_down=0.1):
        assert isinstance(secret_key, str)
        assert isinstance(engine, str)
        assert hasattr(openai, function)
        assert isinstance(max_retry, int) or max_retry is None
        assert isinstance(cool_down, (float, int)) and cool_down > 0.
        self._api = getattr(openai, function)
        self._key = openai.api_key = secret_key
        self._model = engine
        self._retry = max_retry
        self._cool = cool_down
        self._lock = filelock.FileLock(STORED_FILE + ".lock") if STORED_FILE else None

    def __call__(self, *args, **kwrds):
        inputs = self.preprocess(*args, **kwrds)
        internals = self.create(**inputs)
        if self._lock:
            store = json.dumps({"INPUTS": inputs, "OUTPUTS": internals, "TIME": time.asctime()})
            with self._lock:
                with open(STORED_FILE, "a+") as f:
                    f.write(store + '\n')
        return self.postprocess(internals)

    def batch_call(self, queries, batch_size=None, workers=None):
        results = []
        pool = synchronize(self.__call__, queries, batch_size, workers)
        for batch_result in pool:
            results.extend(batch_result)
        return results
        
    def create(self, *args, **kwrds):
        tries = 0
        report = False
        while True:
            try:
                return self._api.create(model=self._model, *args, **kwrds)
            except openai.error.Timeout:
                if tries == self._retry:
                    print("Check you internet please!")
                    return False
            except Exception as e:
                if not report:
                    print(("Unkown Error: %s" % e)[:50])
                    report = True
                    
            time.sleep(self._cool)
            tries += 1

    def preprocess(self, **kwrds):
        return kwrds

    def postprocess(self, outputs):
        return outputs


class Instruct(_APISetup):
    def __init__(self, secret_key, model="text-babbage-001", instruction=None, temperature=1.0, top_p=0.1, n=1):
        _APISetup.__init__(self, secret_key, model, "Completion")
        self._params = {"temperature": temperature, "top_p": top_p, "n": n}
        self.instruction = instruction
        
    @property
    def instruction(self):
        return self._instruct

    @instruction.setter
    def instruction(self, prompt):
        assert isinstance(prompt, str) or prompt is None
        self._instruct = [prompt] if prompt else ['']

    def preprocess(self, new_query):
        prompt = []
        prompt.extend(self._instruct)
        prompt.append(new_query.strip())
        prompt = "\n\n".join(prompt).strip()
        inputs = {"prompt": prompt, "max_tokens": max(0, 1800 - len(prompt.split()))}
        return inputs | self._params

    def postprocess(self, response):
        if response is False:
            return False
        return [_["text"] for _ in response["choices"]]

    @classmethod
    def Babbage(cls, secret_key, instruction=None, temperature=0.0, top_p=0.1, n=1):
        return cls(secret_key, "text-babbage-001", instruction, temperature, top_p, n)

    @classmethod
    def Davinci(cls, secret_key, instruction=None, temperature=1.0, top_p=1.0, n=1):
        return cls(secret_key, "text-davinci-003", instruction, temperature, top_p, n)



class Chatting(_APISetup):
    def __init__(self, secret_key, model="gpt-3.5-turbo-0301", instruction=None, examples=None, cache=False, temperature=1.0, top_p=0.1, n=1):
        _APISetup.__init__(self, secret_key, model, "ChatCompletion")
        self._params = {"temperature": temperature, "top_p": top_p, "n": n}
        self.instruction = instruction
        self.examples = examples
        self._history = [] if cache else None

    @property
    def instruction(self):
        return self._instruct

    @instruction.setter
    def instruction(self, prompt):
        assert isinstance(prompt, str) or prompt is None
        self._instruct = []
        if prompt is not None:
            self._instruct.append({"role": "system", "content": prompt})

    @property
    def examples(self):
        return self._examples

    @examples.setter
    def examples(self, samples):
        if samples is None:
            samples = []
        if isinstance(samples, str):
            samples = [samples]
            
        new_examples = []
        for sample in samples:
            assert len(sample) == 2 and all(map(lambda _: isinstance(_, str), sample)), "each sample has two string terms."
            new_examples.append({"role": "system", "name": "example_user", "content": sample[0]})
            new_examples.append({"role": "system", "name": "example_assistant", "content": sample[1]})
        self._examples = new_examples

    def preprocess(self, new_query):
        new_query = {"role": "user", "content": new_query}
        inputs = []
        inputs.extend(self._instruct)
        inputs.extend(self._examples)
        if self._history is not None:
            inputs.extend(self._history)
            self._history.append(new_query)
        return {"messages": inputs + [new_query]} | self._params

    def postprocess(self, response):
        if response is False:
            if self._history:
                self._history.pop(-1)
            return False
        out_text = [_["message"]["content"] for _ in response["choices"]]
        if self._history:
            self._history.append({"role": "assistant", "content": out_text[0]})
        return out_text

    def clear_history(self):
        self._history.clear()

    def edit_history(self, text, idx=-1):
        assert len(self._history) > 0
        self._history[idx]["content"] = text

    @classmethod
    def ChatGPT(cls, secret_key, instruction=None, examples=None, cache=False, temperature=0.0, top_p=1.0, n=1):
        return cls(secret_key, "gpt-3.5-turbo-0613", instruction, examples, cache, temperature, top_p, n)

    @classmethod
    def GPT4(cls, secret_key, instruction=None, examples=None, cache=False, temperature=0.0, top_p=0.1, n=1):
        return cls(secret_key, "gpt-4-0314", instruction, examples, cache, temperature, top_p, n)


def NeuronInterpretor(key):
    instruct = "You are a neuron interpreter for neural networks. Each neuron looks for one particular concept/topic/theme/behavior/pattern. " +\
               "Look at some words the neuron activates for and summarize in a single concept/topic/theme/behavior/pattern what the neuron is looking for. " +\
               "Don't list examples of words and keep your summary as concise as possible. " +\
               "If you cannot summarize more than half of the given words within one clear concept/topic/theme/behavior/pattern, you should say 'Cannot Tell'."
    examples = [
                ("Words: January, terday, cember, April, July, September, December, Thursday, quished, November, Tuesday.",
                 "dates."),
                ("Words: B., M., e., R., C., OK., A., H., D., S., J., al., p., T., N., W., G., a.C., or, St., K., a.m., L..",
                 "abbrevations and acronyms."),
                ("Words: actual, literal, real, Real, optical, Physical, REAL, virtual, visual.",
                 "perception of reality."),
                ("Words: Go, Python, C++, Java, c#, python3, cuda, java, javascript, basic.",
                 "programing languages."),
                ("Words: 1950, 1980, 1985, 1958, 1850, 1980, 1960, 1940, 1984, 1948.",
                 "years"),
                ]
    return Chatting.ChatGPT(key, instruct, examples, temperature=1.0, top_p=0.9)



def TaskInterpretor(key):
    instruct = "Which of the following assistant tasks can the given concept is used for?\n\nTasks: daily writing, literary writing, professional writing, solving math problems, coding, translation. Return 'None' if it cannot be used for any of the above tasks. If it could be used for multiple tasks, list all of them and seperate with ';'."
    examples = [
            ("Concept: Words are social media post tags.",
             "daily writing"),
            ("Concept: Words are Latex code for drawing a grouped barchart.",
             "professional writing"),
            ("Concept: Words are foreign words or names.",
             "translation"),
            ("Concept: Words are numbers.",
             "solving math problems"),
            ("Concept: Words are URLs.",
             "None"),
            ("Concept: Words are Words related to configuration files and web addresses.",
             "coding"),
            ("Concept: Words are rhyming words.",
             "literary writing"),
            ("Concept: Words are programming commands and terms.",
             "coding")
            ]
    return Chatting.ChatGPT(key, instruct, examples, temperature=0.0, top_p=0.9)



def ConceptInterpretor(key):
    instruct = "You are a linguist. Classify the provided concept into one of the following categories: Phonology, Morphology, Syntax, and Semantic."
    examples = [
            ("Concept: Words are dates.",
             "Semantic"),
            ("Concept: Words are perception of reality.",
             "Semantic"),
            ("Concept: Words are abbrevations and acronyms.",
             "Morphology"),
            ("Concept: Words are related to actions or activities.",
             "Syntax"),
            ("Concept: words are medical abbrivations.",
             "Semantic"),
            ("Concept: Words are URLs.",
             "Morphology"),
            ("Concept: Words are verbs.",
             "Syntax"), 
            ("Concept: Words are adjective.",
             "Syntax"),
            ("Concept: Words are rhyming words.",
             "Phonology"),
            ("Concept: Words are programming languges.",
             "Semantic")
            ]
    return Chatting.ChatGPT(key, instruct, examples, temperature=0.0, top_p=0.9)


def NeuronInterpretation(key, src_file, tgt_file, nwords=15, batchsize=64):
    print("Reading File: %s" % src_file)
    print("Saving File: %s" % tgt_file)
    def interpret_batch(infos, tokens, writeto):
        explains = model.batch_call(tokens, batch_size=len(tokens), workers=32)
        for token, expl, info in zip(tokens, explains, infos):
            expl = expl[0]
            explainable = "0" if "cannot tell" in expl.lower() else "1"
            token = token.replace("\n", "/n").replace("\t", "/t")
            writeto.write("\t".join([explainable, token, expl.replace("\n", "").replace("\t", ""), info]))
        infos.clear()
        tokens.clear()

    model = NeuronInterpretor(key)
    with open(tgt_file, "w", encoding="utf8") as g,\
         open(src_file, "r", encoding="utf8") as f:
        headline = f.readline()
        if g.tell() == 0:
            g.write("Explainable\tTokens\tSummary\t" + headline)
        begin = f.tell()
        for idx, row in enumerate(f, 1):
            pass
        f.seek(begin)
        tokens, infos = [], []
        for ids, row in enumerate(tqdm.tqdm(f, total=idx)):
            if row[:-1].count("\t") != 7:
                print("Row=%d is failed! %s" % (ids, row))
                continue
            time.sleep(0.001)
            name, size, layer, type, head, rank, val, topK = row[:-1].split("\t")
            if name in ["vicuna", "llama", "mistral", "mistral_inst"]:
                infos.append(row)
                topK = topK.replace("/t", "\t").replace("/n", "\n")
                tokens.append("Words: %s." % ", ".join(topK.split("|||")[:nwords]))
            if len(tokens) == batchsize:
                interpret_batch(infos, tokens, g)
        if len(tokens) > 0:
            interpret_batch(infos, tokens, g)


def ConceptInterpretation(key, src_file, tgt_file, batchsize=128):
    print("Reading File: %s" % src_file)
    print("Saving File: %s" % tgt_file)
    def interpret_batch(infos, tokens, writeto):
        explains = model.batch_call(tokens, batch_size=len(tokens), workers=32)
        for token, expl, info in zip(tokens, explains, infos):
            writeto.write("\t".join([expl[0].replace("\n", "").replace("\t", ""), info]))
        infos.clear()
        tokens.clear()

    model = ConceptInterpretor(key)
    with open(tgt_file, "w", encoding="utf8") as g,\
         open(src_file, "r", encoding="utf8") as f:
        headline = f.readline()
        if g.tell() == 0:
            g.write("Concept\t" + headline)
        begin = f.tell()
        for idx, row in enumerate(f, 1):
            pass
        f.seek(begin)
        tokens, infos = [], []
        for ids, row in enumerate(tqdm.tqdm(f, total=idx)):
            if row[:-1].count("\t") != 10:
                print("Row=%d is failed! %s" % (ids, row))
                continue
            time.sleep(0.01)
            explainable, words, summary, name, size, layer, type, head, rank, val, topK = row[:-1].split("\t")
            if name in ["vicuna", "llama", "mistral", "mistral_inst"] and "cannot tell" not in summary.lower():
                infos.append(row)
                tokens.append("Concept: Words are %s" % summary)
            if len(tokens) == batchsize:
                interpret_batch(infos, tokens, g)
        if len(tokens) > 0:
            interpret_batch(infos, tokens, g)


def TaskInterpretation(key, src_file, tgt_file, batchsize=128):
    print("Reading File: %s" % src_file)
    print("Saving File: %s" % tgt_file)
    def interpret_batch(infos, tokens, writeto):
        explains = model.batch_call(tokens, batch_size=len(tokens), workers=32)
        for token, expl, info in zip(tokens, explains, infos):
            writeto.write("\t".join([expl[0].replace("\n", "").replace("\t", ""), info]))
        infos.clear()
        tokens.clear()

    model = TaskInterpretor(key)
    with open(tgt_file, "w", encoding="utf8") as g,\
         open(src_file, "r", encoding="utf8") as f:
        headline = f.readline()
        if g.tell() == 0:
            g.write("Task\t" + headline)
        begin = f.tell()
        for idx, row in enumerate(f, 1):
            pass
        f.seek(begin)
        tokens, infos = [], []
        for ids, row in enumerate(tqdm.tqdm(f, total=idx)):
            if row[:-1].count("\t") != 10:
                print("Row=%d is failed! %s" % (ids, row))
                continue
            time.sleep(0.01)
            explainable, words, summary, name, size, layer, type, head, rank, val, topK = row[:-1].split("\t")
            if name in ["vicuna", "llama", "mistral", "mistral_inst"] and "cannot tell" not in summary.lower():
                infos.append(row)
                tokens.append("Concept: Words are %s" % summary.lower())
            if len(tokens) == batchsize:
                interpret_batch(infos, tokens, g)
        if len(tokens) > 0:
            interpret_batch(infos, tokens, g)


if __name__ == "__main__":
    KEY = "XXX" 
    seed = int(sys.argv[1]) 
    NeuronInterpretation(KEY, "../results/linear_interpret/linear_words.tsv",
                         "../results/linear_interpret/linear_seed%d.tsv" % seed)
    TaskInterpretation(KEY, "../results/linear_interpret/linear_seed%s.tsv" % seed,
                       "../results/linear_interpret/linear_task_seed%s.tsv" % seed)
    ConceptInterpretation(KEY, "../results/linear_interpret/linear_seed%s.tsv" % seed,
                         "../results/linear_interpret/linear_concept_seed%s.tsv" % seed)
        
        

