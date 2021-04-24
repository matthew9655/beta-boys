# How to run the GloVe Encoding

In the GloVe file, run the following command:

```console
python3 glove.py
```

Once this code finishes training the embeddings, it will save three objects into pt files.
They are idx.pt, model.pt, loss.pt. They correspond to word_to_ix, model and losses in the code. 

Next time, to play with the model, just uncomment the loading functions in glove.py

The decoder is for after training the VAE on the encoding, mapping the word vectors to the original embeddings.