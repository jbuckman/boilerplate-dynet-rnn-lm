# RNN Language Model Boilerplate

This is boilerplate code for quickly and easily getting experiments on language modeling off of the ground. The code is written in the Python version of the [DyNet framework](https://github.com/clab/dynet), which can be installed using [these instructions](http://dynet.readthedocs.io/en/latest/python.html).

## Quickstart

To train a baseline Char-RNN model, on any file:

`python train.py --train=<filename> --reader=generic_char --split_train`

From there, there's a bunch of flags you can set to adjust the size of the model, the dropout, the architecture, and many other things. The flags should be pretty self explanatory. You can list the flags with `python train.py -h`

## Saving & Loading Models

If you want to save off a trained model and come back to it later, just use the `--save=FILELOC` flag. Then, you can load it later on with the `load=FILELOC` flag. NOTE: if you load a model, you also load its parameter settings, so the `--load` flag overrides things like `--size`, `--gen_layers`, `--gen_hidden_dim`, etc.

## Choose Training Corpus

By default, this code is set up to train on the Penn Treebank data. I'm not sure if I'm allowed to distribute it publicly, so the folder is currently empty, but if you get your hands on the data, just rename the files to `train.ptb`, `valid.ptb`, and `test.ptb` and put them in the `ptb/` folder.

To add a new data source, simply implement a new CorpusReader in util.py. Make sure that you set the `names` property to be a list that includes at least one unique ID. Then, set the `--reader=ID`, and use the `--train`, `--valid`, and `--test` flags to point to your data set. If you don't have pre-separated data, just set `--train` and include the `--split_train` flag to have your data automatically separated into train, valid, and test splits.

## Implementing New RNN Architectures

To implement a new model, simply go into rnnlm.py, create a new subclass of SaveableRNNLM which implements the functions `add_params`, `BuildLMGraph`, and `BuildLMGraph_batch`. An example is included. Make sure you set the `name` property of your new class to a unique ID, and then use the `--arch=ID` flag to tell the code to use your new model.

## Visualize Logs

To get a clean graph of how your model is training over time, call `python visualize_log.py <filename> <filename>...` to plot up to 20 training runs. The to generate the logfiles used as input for the visualizer, simply include the `--output=<filename>` flag when training.

## Example Use Case

Let's say we wanted to test out [how reuse of word embeddings affects the performance of a language model] (https://openreview.net/pdf?id=r1aPbsFle). We'll be using the PTB corpus, so I don't need to worry about setting up a new corpus reader. First, let's train a baseline model for 10 epochs:

`python train.py --dynet-mem 3000 --word_level --size=small --minibatch_size=24 --save=small_baseline.model --output=small_baseline.log`

`python train.py --dynet-mem 3000 --word_level --minibatch_size=24 --load=small_baseline.model --evaluate`

`>> [Test TEST]     Loss: 4.49591021467     Perplexity: 121.516469934       Time: 17.0475099087`

(This is much worse than the perplexity reported in the paper, but that's because we are just using a generic LSTM model as our baseline, rather than the more complex VD-LSTM model.)

Next, let's modify our baseline language model to incorporate reuse of word embeddings. As an example, I've done this at the bottom of rnnlm.py, creating a class called `ReuseEmbeddingsRNNLM` with `name = "reuse_emb"`. The code is pretty much just a copy-and-paste of the baseline model above, changing around 10 lines to incorporate resuse of word embeddings into the prediction of outputs. Now, let's run those tests too:

`python train.py --dynet-mem 3000 --arch=reuse_emb --word_level --size=small --minibatch_size=24 --save=small_reuseemb.model --output=small_baseline.log`

`python train.py --dynet-mem 3000 --arch=reuse_emb --word_level --minibatch_size=24 --load=small_reuseemb.model --evaluate`

`>> [Test TEST]     Loss: 4.34946627675     Perplexity: 115.64385445        Time: 15.4759089947`

And there we have it - reuse of embeddings gives us a 6-point decrease in perplexity. Very straightforward!

We can visualize this using the included `visualize_log.py` script:

`python visualize_log.py small_baseline.log small_baseline.log`

Producing:

![image of graph](https://github.com/jbuckman/boilerplate-dynet-rnn-lm/blob/master/compare_baseline_reuseemb.png)

## Contact

If you have any questions, please feel free to hit me up: jacobbuckman [.a.t.] cmu.edu
