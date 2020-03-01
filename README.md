# Sentence-similarity-in-the-same-corpus-using-GUSE

This code represernts the use of the [**Google Universal Sentence Encoder**](https://tfhub.dev/google/universal-sentence-encoder/4) to detect similarity among text sentences which might as well be the key in enhancing long tail keyword search.
The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering, and other natural language tasks. The pre-trained Universal Sentence Encoder is publicly available in [Tensorflow-hub](https://tfhub.dev/). 

Here we provide a text file having input sentences for which we want to check the best similarity with other sentences in the same corpus. It outputs similarity among them in the range of 0 to 1, where *1 shows the maximum similarity*(generally in case of exactly same sentences but here those have been removed from the results).

We pick the top 10 similar sentences for each sentence ,for example, here with an input file having sentences with different variants of excavators. This was done to find the best related categories for creating appropriate product mapping hierarchy.


