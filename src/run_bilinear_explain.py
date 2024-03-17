import string
import sys
import pickle as pkl
import os


from generator import Generator
from utils import load_wiki_words, merge_clean


MODELS = {"vicuna": ("7B", "/data/weights/vicuna/hf_models/7B/"),
          "llama": ("7B", "/data/weights/llama/hf_models/7B/"),
          "mistral": ("7B", "mistralai/Mistral-7B-v0.1"),
          "mistral_inst": ("7B", "mistralai/Mistral-7B-Instruct-v0.1")}


def parsing_arguments():
    parser = argparse.ArgumentParser(prog="GIA")
    parser.add_argument("--vocab")
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parsing_arguments()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    words = load_wiki_words(args.vocab) 
    print("Loading %d words from %s" % (len(words), args.vocab))
    print("Saving neuron pairs to: %s" % args.output)
    assert args.model in MODELS
    print("Interpreting SelfAtt layers from %s" % args.name)

    with open(args.output, "a+", encoding="utf8") as f:
        if f.tell() == 0:
            f.write("Model\tSize\tLayer\tType\tHead\tDim\tRank\tScore\tTopK\n")
        size, repo = MODELS[args.model]
        model = Generator(repo)
        for (layer, head, pairs, scores, dims) in model.att_explain(words):
            for rank, (p, s, d) in enumerate(zip(pairs, scores, dims), 1):
                f.write("\t".join([args.model, size, str(layer), "QK", str(head), str(d), 
                                   str(rank), "%.4f" % s, merge_clean(p)]) + '\n')


