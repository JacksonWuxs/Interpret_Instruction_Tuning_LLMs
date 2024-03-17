#### Introduction

This is the official codebase of our paper: [From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning](https://arxiv.org/abs/2310.00492). In this repo, we implement several explanation methods for LLMs, including a gradient-based attribution method, a word-activation-based method for Self-Attention layers, and a weight decomposition method for Feedforward layers. We also provide the analysis scripts to use the explanations to understand the behavior shifts of LLMs after instruction tuning. Currently, this repo also includes the generated explanations for Vicuna-7b-v1.1 and LLaMA-7b. The explanations for other families will be coming soon, such as Mistral.

#### Environment

We assume that you manage the environment with Conda library.

```shell
>>> conda create -n UsableXAI python=3.9 -y
>>> conda activate UsableXAI
>>> pip install -U requirements.txt
>>> cd src
```

#### Explanation Methods

The three explanation methods for our experiments are implemented as three functions of the class `Generator` from the file `./src/generator.py`. Our implementations could be easily extended to other language model families that are available from Huggingface `transformers` library.

* __Input-Output Attribution:__ `Generator.input_explain()` is a local explanation method, which measures the contribution of each input token to the output token(s) normalized and sparsified with the strategy proposed in the paper. _Our latest project shows that this attribution explanations can be used to detect hallucination responses or verify the response quality!!!_ See details at [here](https://github.com/JacksonWuxs/UsableXAI_LLM).
* __Self-Attention Pattern Explain:__ ``Generator.att_explain()`` is a global explanation method, which finds the word-word pairwise patterns to interpret the behavior of each self-attention head under the "local co-occurrence" constraints. 
* __Feedforward Concept Explain:__ ``Generator.ffn_explain()`` is a global explanation method, which analyze the spread of vectors in the feedforward layer by using the Principal Component Analysis (PCA). The main spreading directions (principal components) are then projected to the static word embeddings to achieve concept-level explanations. 

We also include a script to parallelly call ChatGPT APIs for machine annotations in  `./src/run_chatgpt_annotate.py`.

#### Explanation Results

We include the explanation results of Vicuna-7b-v1.1 and LLaMA-7b in the folder `./results/`. We are running experiments on Mistral family and we will update the results within the weeks. 

* __Attribution Explanations:__ Folder`./results/attribution_interpret/ifa_vicuna_7b/` includes 632 cases for Vicuna-7B and folder `./results/attribution_interpret/ifa_llama_7b/` is for LLaMA-7B.
* __Self-Attention Pattern Explanations:__ The results of both models are expected in ONE `.tsv` file in a format of ``Model\tSize\tLayer\tType\tHead\tDim\tRank\tScore\tTopK``, where TopK is a list of word pairs to interpret this dimension of the head. Each word pair looks like `word_1=word_2`, and word pairs are separated with symbol `|||`. We break the whole file into smaller ones to match the Github requirement. To reconstruct the result file, you need to run ``cat bilinear_words_split.tsv* > bilinear_words.tsv`` in the folder `./results/bilinear_interpret/`. 
* __Feedforward Concept Explanations:__ The original result of both models is in the file `./results/linear_interpret/linear_words.tsv` in a format of `Model\tSize\tLayer\tType\tHead\tRank\tScore\tTopK`, where TopK is a list of words (separated with `|||`) to interpret the corresponding principal component. We also include the ChatGPT annotate data with different seed in the same folder. 

#### Reproduction

* __Whether important density demonstrates instruction following ability?__ (Table 1)

  we first need to collect the attribution scores between inputs (prompt) and outputs (responses) for our target model, such as Vicuna-7B. _(You don't have to run this code for reproduction since we have included results in this repo.)_

  ```shell
  >>> nohup python -u run_instruct_following.py --dataset ../datasets/FinalDataset.tsv --model /data/weights/vicuna_7b/ --output ../results/attribution_interpret/ifa_vicuna_7b/ --cuda 0,1 &
  ```

  In addition, we need to provide a datafile manually annotated the quality of each generated response from Vicuna. Also, we need a dataset that denotes which part of the input prompt referring to the instruction. We provide them in `./datasets/AnnotatedDataset.tsv` and `./datasets/FinalDataset.tsv` respectively. You may modify the config inside to test on your own data.

  ```shell
  >>> python analysis_following_hypothesis.py 
  ```

  Then you can see the results that _a higher important density of instruction words over followed samples than unfollowed samples._

* __Whether instruction-tuned models perform better than the pre-trained ones in identifying instructions?__ (Table 2)

  Again, you need to collect the attribution scores on a pre-trained model, such as LLaMA-7b. _(You don't have to run this code for reproduction since we have included results in this repo.)_

  ```shell
  >>> nohup python -u run_instruct_following.py --dataset ../datasets/FinalDataset.tsv --model /data/weights/llama_7b/ --output ../results/attribution_interpret/ifa_llama_7b/ --cuda 0,1 &
  ```

  Then you can check the differences with the following code.

  ```
  >>> python analysis_following_ability_after_IFT.py
  ```

  You can see that _the tuned model better recognizes the instruction words from the input prompts than its pre-trained counterpart._ 

* __Whether self-attention heads encode more instruction verbs after instruction tuning?__ (Table 3)

  We first collect the explanations of our target models, i.e., Vicuna-7B and LLaMA-7B. The weights of these two models are listed in the file `run_bilinear_explain.py` already. You may modify the addresses to your own path. _(You don't have to run this code for reproduction since we have included results in this repo.)_

  ```shell
  >>> nohup python -u run_bilinear_explain.py --vocab ../datasets/SingleShareGPT_Vocab.tsv --model vicuna --output ../results/bilinear_interpret/bilinear_words.tsv --cuda 0,1 &
  >>> nohup python -u run_bilinear_explain.py --vocab ../datasets/SingleShareGPT_Vocab.tsv --model llama --output ../results/bilinear_interpret/bilinear_words.tsv --cuda 0,1 &
  ```

  We then can tract the change of occurrent frequency of certain words (instruction verbs and frequent verbs) after instruction tuning. 

  ```shell
  >>> python analysis_bilinear_interpret.py ../results/bilinear_interpret/bilinear_words.tsv
  ```

  You can see that _the instruction verbs are more common in the instruction-tuned models than the pre-trained ones, while general frequent verbs don't share this trend._

* __Whether feedforward network rotates its pre-trained knowledge after instruction tuning?__ (Table 5 and Figure 3)

  We first collect the explanations of our target models, i.e., Vicuna-7B and LLaMA-7B. The weights of these two models are listed in the file `run_bilinear_explain.py` already. You may modify the addresses to your own path. _(You don't have to run this code for reproduction since we have included results in this repo.)_

  ```shell
  >>> nohup python -u run_linear_explain.py --vocab ../datasets/SingleShareGPT_Vocab.tsv --model vicuna --output ../results/linear_interpret/linear_words.tsv --cuda 0,1 &
  >>> nohup python -u run_linear_explain.py --vocab ../datasets/SingleShareGPT_Vocab.tsv --model llama --output ../results/linear_interpret/linear_words.tsv --cuda 0,1 &
  ```

  We then leverage ChatGPT to annotate the linguistic level and suitable tasks for each concept according to the word lists. This process involves two wo steps: (1) generating concept description according to the word lists, and (2) classifying linguistic knowledge or suitable tasks according to the concepts. We will conduct this process over 5 random seeds.  Please input your own OpenAI API Token in `run_chatgpt_annotate.py` file. _(You don't have to run this code for reproduction since we have included results in this repo.)_

  ```shell
  # running seeds 0, 1, 2, 3, 4
  # the ChatGPT version we used: gpt-3.5-turbo-0613
  >>> nohup python run_chatgpt_annotate.py 0
  >>> nohup python run_chatgpt_annotate.py 1
  >>> nohup python run_chatgpt_annotate.py 2
  >>> nohup python run_chatgpt_annotate.py 3
  >>> nohup python run_chatgpt_annotate.py 4
  ```

  Now, we can check the change of linguistic disciplines or suitable tasks after instruction tuning.

  ```shell
  >>> python analysis_linear_interpret.py ../results/linear_interpret/linear_
  ```

  You can see the change of percentage of knowledge between Pre-trained and Instruction-tuned models categorized by linguistic disciplines or downstream tasks. It will also generate a figure in `ProportionOfLinguisticKnowledge.pdf` to show the percentage of knowledge categorized by linguistic disciplines across different layers. 

#### Citations

If you use code or data from this repo, please cite our paper as followed.

```latex
@article{wu2023language,
  title={From language modeling to instruction following: Understanding the behavior shift in llms after instruction tuning},
  author={Wu, Xuansheng and Yao, Wenlin and Chen, Jianshu and Pan, Xiaoman and Wang, Xiaoyang and Liu, Ninghao and Yu, Dong},
  journal={arXiv preprint arXiv:2310.00492},
  year={2023}
}
```

