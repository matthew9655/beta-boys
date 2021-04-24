# Sentence Variational Autoencoder

## The code mainly builds off of [timbmg's repository](https://github.com/timbmg/Sentence-VAE) which is a reimplementation of the [paper](https://arxiv.org/pdf/2006.00693.pdf) 

## Training
To run the training, please download the Penn Tree Bank data first (download from [Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)). The code expects to find at least `ptb.train.txt` and `ptb.valid.txt` in the specified data directory. The data can also be donwloaded with the `dowloaddata.sh` script.

Then training can be executed with the following command:
```
python3 train.py
```

The following arguments are available:

`--data_dir`  The path to the directory where PTB data is stored, and auxiliary data files will be stored.  
`--create_data` If provided, new auxiliary data files will be created form the source data.  
`--max_sequence_length` Specifies the cut off of long sentences.  
`--min_occ` If a word occurs less than "min_occ" times in the corpus, it will be replaced by the <unk> token.  
`--test` If provided, performance will also be measured on the test set.

`-ep`, `--epochs`  
`-bs`, `--batch_size`  
`-lr`, `--learning_rate`

`-eb`, `--embedding_size`  
`-rnn`, `--rnn_type` Either 'rnn' or 'gru'.  
`-hs`, `--hidden_size`  
`-nl`, `--num_layers`  
`-bi`, `--bidirectional`  
`-ls`, `--latent_size`  
`-wd`, `--word_dropout` Word dropout applied to the input of the Decoder, which means words will be replaced by `<unk>` with a probability of `word_dropout`.  
`-ed`, `--embedding_dropout` Word embedding dropout applied to the input of the Decoder.

`-af`, `--anneal_function` Either 'logistic' or 'linear'.  
`-k`, `--k` Steepness of the logistic annealing function.  
`-x0`, `--x0` For 'logistic', this is the mid-point (i.e. when the weight is 0.5); for 'linear' this is the denominator.

`-v`, `--print_every`  
`-tb`, `--tensorboard_logging` If provided, training progress is monitored with tensorboard.  
`-log`, `--logdir` Directory of log files for tensorboard.  
`-bin`,`--save_model_path` Directory where to store model checkpoints.

## Inference
For obtaining samples and interpolating between senteces, inference.py can be used.
```
python3 inference.py -c $CHECKPOINT -n $NUM_SAMPLES
```

## Latent Traversals
To obtain latent traversals,
```
python3 latent_traverse.py -c $CHECKPOINT 
```

