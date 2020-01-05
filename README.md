# KL-divergence-sentence-similarity
Use KL-Divergence to find the relavance scores or relative importance of candidate sentences in a large corpus of documents. 
We can also use this to find the similarity between two sentences in a large corpus of documents. 
We do this by first computing the distribution of words in each candidate sentence and also the distribution of words in the entire corpus of documents.
Finally, we use KL-Divergence to calculate the similarity between the given two distributions. Lower the KL-divergence value, greater is the similarity/relevance of the candidate sentence to the corpus.
Similarily, higher the KL-divergence between the distributions of two sentences, lesser  is the similarity between them and vice-versa.
