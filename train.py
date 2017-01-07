import dynet
import util
import rnnlm as rnnlm
import argparse, random, time, sys, math
random.seed(78789) # I like setting a seed for consistent behavior when debugging

parser = argparse.ArgumentParser()

## need to have this dummy argument for dynet
parser.add_argument("--dynet-mem")
parser.add_argument("--dynet-gpu")

## locations of data
parser.add_argument("--train", default="ptb/train.ptb")
parser.add_argument("--valid", default="ptb/valid.ptb")
parser.add_argument("--test", default="ptb/test.ptb")
parser.add_argument("--reader")

## alternatively, load one dataset and split it
parser.add_argument("--split_train", action='store_true')
parser.add_argument("--percent_valid", default=1000, type=float)
parser.add_argument("--percent_test", default=.05, type=float)

## vocab parameters
parser.add_argument('--rebuild_vocab', action='store_true')
parser.add_argument('--unk_thresh', default=5, type=int)

## rnn parameters
parser.add_argument("--size", choices={"small", "medium", "large", "enormous"})
parser.add_argument("--gen_layers", default=1, type=int)
parser.add_argument("--gen_input_dim", default=10, type=int)
parser.add_argument("--gen_hidden_dim", default=50, type=int)
parser.add_argument("--rnn", default="lstm")
parser.add_argument("--dropout", default=.1, type=float)

## experiment parameters
parser.add_argument("--trainer", default="adam", choices={"sgd", "adam", "adagrad"})
parser.add_argument("--learning_rate")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--minibatch_size", default=1, type=int)
parser.add_argument("--unbatch_idx", default=50, type=int)
parser.add_argument("--log_train_every_n", default=100, type=int)
parser.add_argument("--log_valid_every_n", default=5000, type=int)
parser.add_argument("--output")
parser.add_argument("--begin_token", default='<s>')
parser.add_argument("--end_token", default='<e>')

## choose what model to use
parser.add_argument("--arch", default="baseline")
parser.add_argument("--save")
parser.add_argument("--load")

## other parameters
parser.add_argument("--word_level", action="store_true")
parser.add_argument("--evaluate", action="store_true")

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
    if args.word_level: CORPUS_READ_STYLE = "ptb_stripped"
    else: CORPUS_READ_STYLE = "ptb_char_stripped"
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
    vocab.add_unk(args.unk_thresh)
    lm = rnnlm.get_model(args.arch)(model, vocab, args)

################################### LOAD THE DATA
train_data = list(util.get_reader(CORPUS_READ_STYLE)(args.train, mode=CORPUS_READ_STYLE, begin=BEGIN_TOKEN, end=END_TOKEN))
if not args.split_train:
    valid_data = list(util.get_reader(CORPUS_READ_STYLE)(args.valid, mode=CORPUS_READ_STYLE, begin=BEGIN_TOKEN, end=END_TOKEN))
    test_data  = list(util.get_reader(CORPUS_READ_STYLE)(args.test, mode=CORPUS_READ_STYLE, begin=BEGIN_TOKEN, end=END_TOKEN))

################################### SPLIT THE DATA (IF NEEDED)
if args.split_train:
    if args.percent_valid > 1: vc = args.percent_valid
    else: vc = int(len(train_data)*(args.percent_valid))
    if args.percent_test > 1: tc = args.percent_test
    else: tc = int(len(train_data)*(args.percent_test))
    valid_data = train_data[-(vc+tc):-tc]
    test_data = train_data[-tc:]
    train_data = train_data[:-(vc+tc)]

################################### WIPE THE OUTPUT FILE
if args.output:
    outfile = open(args.output, 'w')
    outfile.write("")
    outfile.close()

################################### PREPARE THYSELF FOR EPOCHS
train_data.sort(key=lambda x:-len(x))
valid_data.sort(key=lambda x:-len(x))
test_data.sort(key=lambda x:-len(x))

unbatch_idx = args.minibatch_size*(int(args.unbatch_idx/args.minibatch_size)+1)
train_mb_indices = range(unbatch_idx) + range(unbatch_idx, len(train_data), args.minibatch_size)
# train_mb_indices = range(0, len(train_data), args.minibatch_size)
valid_mb_indices = range(0, len(valid_data), args.minibatch_size)
test_mb_indices = range(0, len(test_data), args.minibatch_size)

predictions_made = lambda x: len(x) - 1

################################### LEGGO
best_score = None
token_count = sent_count = cum_loss = cum_perplexity = 0.0
_start = time.time()
for ITER in range(args.epochs):
    lm.epoch = ITER
    random.shuffle(train_mb_indices)

    for i, index in enumerate(train_mb_indices):
        sample_num = (1+i)*args.minibatch_size+(len(train_data)*ITER)

        #### train logging
        if (sample_num % args.log_train_every_n) / args.minibatch_size == 0:
            print ITER, sample_num, " ",
            trainer.status()
            print "L:", cum_loss / token_count,
            print "P:", cum_perplexity / sent_count,
            print "T:", (time.time() - _start),
            _start = time.time()
            sample = lm.sample(first=BEGIN_TOKEN,stop=END_TOKEN,nchars=1000)
            if sample: print lm.vocab.pp(sample),
            token_count = sent_count = cum_loss = cum_perplexity = 0.0
            print
        #### end of train logging

        #### validation logging
        if (sample_num % args.log_valid_every_n) / args.minibatch_size == 0:
            v_token_count = v_sent_count = v_cum_loss = v_cum_perplexity = 0.0
            v_start = time.time()
            for v_i in valid_mb_indices:
                v_batch = valid_data[v_i:v_i+args.minibatch_size]
                v_isents = [[lm.vocab[w].i for w in v_sent] for v_sent in v_batch]
                if args.minibatch_size == 1:    v_losses = lm.BuildLMGraph(v_isents[0], sent_args={"test":True})
                else:                           v_losses = lm.BuildLMGraph_batch(v_isents, sent_args={"test":True})
                v_gen_losses = v_losses.vec_value()
                v_cum_loss += sum(v_losses.vec_value())
                v_cum_perplexity += sum([math.exp(v_gen_loss / predictions_made(v_sent)) for v_gen_loss, v_sent in zip(v_gen_losses, v_batch)])
                v_token_count += sum([predictions_made(v_sent) for v_sent in v_batch])
                v_sent_count += len(v_batch)
            print "[Validation "+str(sample_num) + "]\t" + \
                  "Loss: "+str(v_cum_loss / v_token_count) + "\t" + \
                  "Perplexity: "+str(v_cum_perplexity / v_sent_count) + "\t" + \
                  "Time: "+str(time.time() - v_start),
            if args.save:
                if best_score is None or best_score > v_cum_perplexity / v_sent_count:
                    print "new best...saving to", args.save
                    best_score = v_cum_perplexity / v_sent_count
                    lm.save(args.save)
            if args.output:
                print "(logging to", args.output + ")"
                with open(args.output, "a") as outfile:
                    outfile.write(str(ITER) + "\t" + \
                                  str(sample_num) + "\t" + \
                                  str(v_cum_loss / v_token_count) + "\t" + \
                                  str(v_cum_perplexity / v_sent_count) + "\n")
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
        cum_perplexity += sum([math.exp(gen_loss / predictions_made(sent)) for gen_loss, sent in zip(gen_losses, sents)])
        token_count += sum([predictions_made(sent) for sent in sents])
        sent_count += len(sents)
        #### end of run training

        loss.backward()
        trainer.update(learning_rate)
        # end of one-sentence train loop
    trainer.update_epoch(learning_rate)
    # end of iteration
# end of training loop

ITER = "TEST"
sample_num = "TEST"
t_token_count = t_sent_count = t_cum_loss = t_cum_perplexity = 0.0
t_start = time.time()
for t_i in test_mb_indices:
    t_batch = test_data[t_i:t_i+args.minibatch_size]
    t_isents = [[lm.vocab[w].i for w in t_sent] for t_sent in t_batch]
    if args.minibatch_size == 1:   t_losses = lm.BuildLMGraph(t_isents[0], sent_args={"test":True})
    else:                          t_losses = lm.BuildLMGraph_batch(t_isents, sent_args={"test":True})
    t_gen_losses = t_losses.vec_value()
    t_cum_loss += sum(t_losses.vec_value())
    t_cum_perplexity += sum([math.exp(t_gen_loss / predictions_made(t_sent)) for t_gen_loss, t_sent in zip(t_gen_losses, t_batch)])
    t_token_count += sum([predictions_made(t_sent) for t_sent in t_batch])
    t_sent_count += len(t_batch)
print "[Test "+str(sample_num) + "]\t" + \
      "Loss: "+str(t_cum_loss / t_token_count) + "\t" + \
      "Perplexity: "+str(t_cum_perplexity / t_sent_count) + "\t" + \
      "Time: "+str(time.time() - t_start),
if args.output:
    print "(logging to", args.output + ")"
    with open(args.output, "a") as outfile:
        outfile.write(str(ITER) + "\t" + \
                      str(sample_num) + "\t" + \
                      str(t_cum_loss / t_token_count) + "\t" + \
                      str(t_cum_perplexity / t_sent_count) + "\n")
