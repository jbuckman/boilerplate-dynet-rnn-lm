This is boilerplate code for quickly and easily getting experiments on language modeling off of the ground. The code is written in the DyNet framework.

To train a model, simply call: `python train.py [FLAGS]`
To evaluate a model, call: `python train.py --evaluate [FLAGS]`
The flags should be pretty self explanatory; they'll be fully documented at some point in the near future.

By default, this code is set up to train on the Penn Treebank data. I'm not sure if I'm allowed to distribute it publicly, so the folder is currently empty, but if you get your hands on the data, just rename the files to `train.ptb`, `valid.ptb`, and `test.ptb` and put them in the `ptb/` folder.

To add a new data source, simply implement a new CorpusReader in util.py. Make sure that you set the `names` property to be a list that includes at least one unique ID. Then, set the `--reader=ID`, and use the `--train`, `--valid`, and `--test` flags to point to your data set. If you don't have pre-separated data, just set `--train` and include the `--split_data` flag to have your data automatically separated into train, valid, and test splits.

To implement a new model, simply go into rnnlm.py, create a new subclass of SaveableRNNLM which implements the functions `add_params`, `BuildLMGraph`, and `BuildLMGraph_batch`. An example is included. Make sure you set the `name` property of your new class to a unique ID, and then use the `--arch=ID` flag to tell the code to use your new model.

If you have any questions, feel free to hit me up: jacobbuckman@cmu.edu
