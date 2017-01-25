# RNN Language Model Boilerplate

This is boilerplate code for quickly and easily getting experiments on language modeling off of the ground. The code is written in the Python version of the [DyNet framework](https://github.com/clab/dynet), which can be installed using [these instructions](http://dynet.readthedocs.io/en/latest/python.html).

It also has [pattern.en](http://www.clips.ua.ac.be/pages/pattern-en) as a dependency for tokenization, if you are using the default reader at word-level.

For an introduction to RNN language models, how they work, and some cool demos, please take a look at Andrej Karpathy's excellent blog post: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

## Quickstart

To train a baseline RNN model on the Penn Treebank:

Char-level: `python train.py`

Word-level: `python train.py --word_level`

To train a baseline RNN model, on any file:

Char-level: `python train.py --train=<filename> --reader=generic_char --split_train`

Word-level: `python train.py --train=<filename> --reader=generic_word --split_train`

From there, there's a bunch of flags you can set to adjust the size of the model, the dropout, the architecture, and many other things. The flags should be pretty self explanatory. You can list the flags with `python train.py -h`

## Saving & Loading Models

If you want to save off a trained model and come back to it later, just use the `--save=FILELOC` flag. Then, you can load it later on with the `load=FILELOC` flag. NOTE: if you load a model, you also load its parameter settings, so the `--load` flag overrides things like `--size`, `--gen_layers`, `--gen_hidden_dim`, etc.

## Choose Training Corpus

By default, this code is set up to train on the Penn Treebank data, which is included in the repo in the `ptb/` folder.

To add a new data source, simply implement a new CorpusReader in util.py. Make sure that you set the `names` property to be a list that includes at least one unique ID. Then, set the `--reader=ID`, and use the `--train`, `--valid`, and `--test` flags to point to your data set. If you don't have pre-separated data, just set `--train` and include the `--split_train` flag to have your data automatically separated into train, valid, and test splits.

## Implementing New RNN Architectures

To implement a new model, simply go into rnnlm.py, create a new subclass of SaveableRNNLM which implements the functions `add_params`, `BuildLMGraph`, and `BuildLMGraph_batch`. An example is included. Make sure you set the `name` property of your new class to a unique ID, and then use the `--arch=ID` flag to tell the code to use your new model.

## Visualize Logs

To get a clean graph of how your model is training over time, call `python visualize_log.py <filename> <filename>...` to plot up to 20 training runs. The to generate the logfiles used as input for the visualizer, simply include the `--output=<filename>` flag when training.

## Scaling Up

The `--size` parameter comes in four settings: small (1 layer, 128 parameters per embedding, 128 nodes in the recurrent hidden layer), medium (2 layers, 256 input dim, 256 hidden dim), large (2, 512, 512), and enormous (2, 1024, 1024). You can also set the parameters individually, with the flags `--gen_layers`, `--gen_input_dim`, and `--gen_hidden_dim`.

## Example Use Case

Let's say we wanted to test out [how reuse of word embeddings affects the performance of a language model] (https://openreview.net/pdf?id=r1aPbsFle). We'll be using the PTB corpus, so we don't need to worry about setting up a new corpus reader - just use the default.

### Train & Evaluate Baseline

First, let's train a baseline model for 10 epochs:

`python train.py --dynet-mem 3000 --word_level --size=small --minibatch_size=24 --save=small_reuseemb.model --output=small_baseline.log`

(Wait for around 2-3 hours)

`python train.py --dynet-mem 3000 --word_level --minibatch_size=24 --load=small_baseline.model --evaluate`

`[Test TEST]     Loss: 5.0651960628      Perplexity: 158.411497631       Time: 20.4854779243`

This is much worse than the baseline perplexity reported in the paper, but that's because we are just using a generic LSTM model as our baseline, rather than the more complex VD-LSTM model, and with many fewer parameters.

### Write, Train & Evaluate Another Model

Next, let's modify our baseline language model to incorporate reuse of word embeddings. As an example, I've done this in rnnlm.py, creating a class called `ReuseEmbeddingsRNNLM` with `name = "reuse_emb"`. The code is pretty much just a copy-and-paste of the baseline model above, changing around 10 lines to incorporate resuse of word embeddings into the prediction of outputs. Now, let's run those tests too:

`python train.py --dynet-mem 3000 --arch=reuse_emb --word_level --size=small --minibatch_size=24 --save=small_reuseemb.model --output=small_reuseemb.log`

(Wait for around 2-3 hours)

`python train.py --dynet-mem 3000 --arch=reuse_emb --word_level --minibatch_size=24 --load=small_reuseemb.model --evaluate`

`[Test TEST]     Loss: 4.88281608367     Perplexity: 132.001869276       Time: 20.2611508369`

And there we have it - reuse of embeddings gives us a 26-point decrease in perplexity.  Nice!

### Visualize Training

Since we turned on the flag for output logs, `--output=small_baseline.log` and `--output=small_reuseemb.log`, we can visualize our validation error over time during training by using the included `visualize_log.py` script:

`python visualize_log.py small_baseline.log small_baseline.log --output=compare_baseline_reuseemb.png`

Producing:

![image of graph](https://github.com/jbuckman/boilerplate-dynet-rnn-lm/blob/master/compare_baseline_reuseemb.png)

## Contact

If you have any questions, please feel free to hit me up: jacobbuckman@cmu.edu
