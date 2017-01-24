import dynet
import util
import rnnlm as rnnlm
import argparse, random, time, sys, math
random.seed(78789) # I like setting a seed for consistent behavior when debugging

parser = argparse.ArgumentParser()

## need to have this dummy argument for dynet
parser.add_argument("--dynet-mem", help="set size of dynet memory allocation, in MB")
parser.add_argument("--dynet-gpu", help="use GPU acceleration")

## locations of data
parser.add_argument("--train", default="ptb/mikolov/train.ptb", help="location of training data")
parser.add_argument("--valid", default="ptb/mikolov/valid.ptb", help="location of validation data")
parser.add_argument("--test", default="ptb/mikolov/test.ptb", help="location of test data")
parser.add_argument("--reader", help="choose which CorpusReader subclass will be used for parsing raw data into tokens")

## alternatively, load one dataset and split it
parser.add_argument("--split_train", action='store_true', help="rather than loading valid & test sets, just split the training data into three pieces")
parser.add_argument("--valid_size", default=1000, type=float, help="if using --split_train, choose the proportion of data to make into validation set." +
                                                                   "if this is less than 1, it is treated as a proportion of the total number of " +
                                                                   "training examples; otherwise, it is a count")
parser.add_argument("--test_size", default=.05, type=float, help="if using --split_train, choose the proportion of data to make into test set." +
                                                                   "if this is less than 1, it is treated as a proportion of the total number of " +
                                                                   "training examples; otherwise, it is a count")

## vocab parameters
parser.add_argument('--rebuild_vocab', action='store_true', help="rebuild the vocabulary rather than using the cached vocabulary")
parser.add_argument('--unk_thresh', default=0, type=int, help="choose the minimum number of times a token needs to appear before being considered an <UNK>")

## rnn parameters
parser.add_argument("--size", choices={"small", "medium", "large", "enormous"}, help="convenience flag for setting the size of the RNN")
parser.add_argument("--gen_layers", default=1, type=int, help="choose number of layers for RNN")
parser.add_argument("--gen_input_dim", default=10, type=int, help="choose token embedding dimension")
parser.add_argument("--gen_hidden_dim", default=50, type=int, help="choose size of hidden state of RNN")
parser.add_argument("--rnn", default="lstm", choices={"lstm","rnn","gru"}, help="choose type of RNN")
parser.add_argument("--dropout", default=.1, type=float, help="set dropout probability")

## experiment parameters
parser.add_argument("--trainer", default="sgd", choices={"sgd", "adam", "adagrad"}, help="choose training algorithm")
parser.add_argument("--learning_rate", help="set learning rate of trainer")
parser.add_argument("--epochs", default=10, type=int, help="maximum number of epochs to run experiment")
parser.add_argument("--minibatch_size", default=1, type=int, help="size of minibatches")
parser.add_argument("--unbatch_idx", default=0, type=int, help="sometimes, with very long inputs, there will be long inputs that are the only input of that length."
                                                               " if these are included in minibatches, we end up using a ton of padding on the other sentnces in the"
                                                               " minibatch to compensate, and this often results in a ton of memory being consumed. this flag sets the"
                                                               " number of sentences in the training set to skip minibatching on - the longest N inputs will be "
                                                               "given batches of size 1")
parser.add_argument("--log_train_every_n", default=100, type=int, help="how often to log training loss")
parser.add_argument("--log_valid_every_n", default=5000, type=int, help="how often to evaluate on validation set, log the loss, and potentially save off the model")
parser.add_argument("--output", help="file location to log validation outputs to")
parser.add_argument("--begin_token", default='<s>', help="special token prepended to sequences")
parser.add_argument("--end_token", default='<e>', help="special token appended to sequences")

## choose what model to use
parser.add_argument("--arch", default="baseline", help="choose what RNNLM architecture you want to use")
parser.add_argument("--save", help="location to save model")
parser.add_argument("--load", help="location to load model from")

## other parameters
parser.add_argument("--word_level", action="store_true", help="if doing LM on PTB, convenience flag for deciding between character and word level")
parser.add_argument("--evaluate", action="store_true", help="convenience flag for runnning just test validation (no train step)")

args = parser.parse_args()
print "ARGS:", args

BEGIN_TOKEN = args.begin_token
END_TOKEN = args.end_token

if args.size == "small":
    args.gen_input_dim = 128
    args.gen_hidden_dim = 128
elif args.size == "medium":
    args.gen_layers = 2
    args.gen_input_dim = 256
    args.gen_hidden_dim = 256
elif args.size == "large":
    args.gen_layers = 2
    args.gen_input_dim = 512
    args.gen_hidden_dim = 512
elif args.size == "enormous":
    args.gen_layers = 2
    args.gen_input_dim = 1024
    args.gen_hidden_dim = 1024

if args.reader is None:
    if args.word_level: CORPUS_READ_STYLE = "generic_word"
    else: CORPUS_READ_STYLE = "generic_char"
else:
    CORPUS_READ_STYLE = args.reader

if args.evaluate: args.epochs = 0

################################### DYNET
model = dynet.Model()
if args.trainer == "sgd":
    trainer = dynet.SimpleSGDTrainer(model)
    learning_rate = 1.0
elif args.trainer == "adam":
    trainer = dynet.AdamTrainer(model)
    learning_rate = .001
elif args.trainer == "adagrad":
    trainer = dynet.AdagradTrainer(model)
    learning_rate = .01

if args.learning_rate is not None: learning_rate = args.learning_rate
################################### LOAD THE MODELS

if args.load:
    lm = rnnlm.get_model(args.arch).load(model, args.load)
    # OVERRIDES
else:
    reader = util.get_reader(CORPUS_READ_STYLE)(args.train, mode=CORPUS_READ_STYLE, begin=BEGIN_TOKEN, end=END_TOKEN)
    vocab = util.Vocab.load_from_corpus(reader, remake=args.rebuild_vocab)
    vocab.START_TOK = vocab[BEGIN_TOKEN]
    vocab.END_TOK = vocab[END_TOKEN]
    if args.unk_thresh > 0: vocab.add_unk(args.unk_thresh, "<UNK>")
    lm = rnnlm.get_model(args.arch)(model, vocab, args)

################################### LOAD THE DATA
train_data = list(util.get_reader(CORPUS_READ_STYLE)(args.train, mode=CORPUS_READ_STYLE, begin=BEGIN_TOKEN, end=END_TOKEN))
if not args.split_train:
    valid_data = list(util.get_reader(CORPUS_READ_STYLE)(args.valid, mode=CORPUS_READ_STYLE, begin=BEGIN_TOKEN, end=END_TOKEN))
    test_data  = list(util.get_reader(CORPUS_READ_STYLE)(args.test, mode=CORPUS_READ_STYLE, begin=BEGIN_TOKEN, end=END_TOKEN))

################################### SPLIT THE DATA (IF NEEDED)
if args.split_train:
    if args.valid_size > 1: vc = args.valid_size
    else: vc = int(len(train_data)*(args.valid_size))
    if args.test_size > 1: tc = args.test_size
    else: tc = int(len(train_data)*(args.test_size))
    valid_data = train_data[-(vc+tc):-tc]
    test_data = train_data[-tc:]
    train_data = train_data[:-(vc+tc)]
    if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
        raise Exception("either your train, validation, or test set is of size 0; adjust --valid_size and --test_size")

################################### WIPE THE OUTPUT FILE
if args.output:
    outfile = open(args.output, 'w')
    outfile.write("")
    outfile.close()

################################### PREPARE THYSELF FOR EPOCHS
train_data.sort(key=lambda x:-len(x))
valid_data.sort(key=lambda x:-len(x))
test_data.sort(key=lambda x:-len(x))

unbatch_idx = args.minibatch_size*(int(args.unbatch_idx/args.minibatch_size))
train_mb_indices = range(unbatch_idx) + range(unbatch_idx, len(train_data), args.minibatch_size)
valid_mb_indices = range(0, len(valid_data), args.minibatch_size)
test_mb_indices = range(0, len(test_data), args.minibatch_size)

words_predicted = lambda x: len(x) - 1

################################### LEGGO
best_score = None
token_count = sent_count = cum_loss = cum_perplexity = 0.0
log_train_counter = args.log_train_every_n
log_valid_counter = args.log_valid_every_n
sample_num = 0
_start = time.time()
for ITER in range(args.epochs):
    lm.epoch = ITER
    random.shuffle(train_mb_indices)

    for i, index in enumerate(train_mb_indices):
        #### train logging
        if log_train_counter <= 0:
            log_train_counter += args.log_train_every_n
            print ITER, sample_num, " ",
            trainer.status()
            print "L:", cum_loss / token_count,
            print "P:", math.exp(cum_loss / token_count),
            print "T:", (time.time() - _start),
            _start = time.time()
            sample = lm.sample(first=BEGIN_TOKEN,stop=END_TOKEN,nchars=10)
            if sample: print lm.vocab.pp(sample),
            token_count = sent_count = cum_loss = cum_perplexity = 0.0
            print
        #### end of train logging

        #### validation logging
        if log_valid_counter <= 0:
            log_valid_counter += args.log_valid_every_n
            v_token_count = v_sent_count = v_cum_loss = 0.0
            v_start = time.time()
            for v_i in valid_mb_indices:
                v_batch = valid_data[v_i:v_i+args.minibatch_size]
                v_isents = [[lm.vocab[w].i for w in v_sent] for v_sent in v_batch]
                if args.minibatch_size == 1:    v_losses = lm.BuildLMGraph(v_isents[0], sent_args={"test":True})
                else:                           v_losses = lm.BuildLMGraph_batch(v_isents, sent_args={"test":True})
                v_gen_losses = v_losses.vec_value()
                v_cum_loss += sum(v_losses.vec_value())
                v_token_count += sum([words_predicted(v_sent) for v_sent in v_batch])
                v_sent_count += len(v_batch)
            v_cum_perplexity = math.exp(v_cum_loss / v_token_count)
            print "[Validation "+str(sample_num) + "]\t" + \
                  "Loss: "+str(v_cum_loss / v_token_count) + "\t" + \
                  "Perplexity: "+str(v_cum_perplexity) + "\t" + \
                  "Time: "+str(time.time() - v_start),

            if args.save:
                if best_score is None or best_score > v_cum_perplexity:
                    print "new best...saving to", args.save
                    best_score = v_cum_perplexity
                    lm.save(args.save)
            if args.output:
                print "(logging to", args.output + ")"
                with open(args.output, "a") as outfile:
                    outfile.write(str(ITER) + "\t" + \
                                  str(sample_num) + "\t" + \
                                  str(v_cum_loss / v_token_count) + "\t" + \
                                  str(v_cum_perplexity) + "\n")
            print "\n"
        #### end of validation logging

        #### run training
        if index < unbatch_idx: sents = train_data[index:index+1]
        else: sents = train_data[index:index+args.minibatch_size]
        isents = [[lm.vocab[w].i for w in sent] for sent in sents]
        if args.minibatch_size == 1: losses = lm.BuildLMGraph(isents[0], sent_args={"test":False})
        else:                        losses = lm.BuildLMGraph_batch(isents, sent_args={"test":False})
        gen_losses = losses.vec_value()
        loss = dynet.sum_batches(losses)
        cum_loss += loss.value()
        cum_perplexity += sum([math.exp(gen_loss / words_predicted(sent)) for gen_loss, sent in zip(gen_losses, sents)])
        token_count += sum([words_predicted(sent) for sent in sents])
        sent_count += len(sents)
        #### end of run training

        loss.backward()
        trainer.update(learning_rate)
        sample_num += len(sents)
        log_train_counter -= len(sents)
        log_valid_counter -= len(sents)
        # end of one-sentence train loop
    trainer.update_epoch(learning_rate)
    # end of iteration
# end of training loop

ITER = "TEST"
sample_num = "TEST"
t_token_count = t_sent_count = t_cum_loss = 0.0
t_start = time.time()
for t_i in test_mb_indices:
    t_batch = test_data[t_i:t_i+args.minibatch_size]
    t_isents = [[lm.vocab[w].i for w in t_sent] for t_sent in t_batch]
    if args.minibatch_size == 1:   t_losses = lm.BuildLMGraph(t_isents[0], sent_args={"test":True})
    else:                          t_losses = lm.BuildLMGraph_batch(t_isents, sent_args={"test":True})
    t_gen_losses = t_losses.vec_value()
    t_cum_loss += sum(t_losses.vec_value())
    t_token_count += sum([words_predicted(t_sent) for t_sent in t_batch])
    t_sent_count += len(t_batch)
t_cum_perplexity = math.exp(t_cum_loss / t_token_count)
print "[Test "+str(sample_num) + "]\t" + \
      "Loss: "+str(t_cum_loss / t_token_count) + "\t" + \
      "Perplexity: "+str(t_cum_perplexity) + "\t" + \
      "Time: "+str(time.time() - t_start),
#       "Perplexity: "+str(t_cum_perplexity / t_sent_count) + "\t" + \
if args.output:
    print "(logging to", args.output + ")"
    with open(args.output, "a") as outfile:
        outfile.write(str(ITER) + "\t" + \
                      str(sample_num) + "\t" + \
                      str(t_cum_loss / t_token_count) + "\t" + \
                      str(t_cum_perplexity) + "\n")
if args.qual:
    print lm.qual_anal()
