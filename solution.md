# solution 
## model

We use a single xgboost model with 290 features are generate from next section.

The exact model is
```python
XGBClassifier(learning_rate=0.05, max_depth=10, subsample=0.8, colsample_bytree=0.8, n_estimators=1000,
                      min_child_weight = 1,
                      nthread=32)
```

We assign weight 3,1,1 to generationMethod 1,2,3 in training which sightly improve score.

The model for final leaderboard is in file leaderboad_solution.py which give public score 0.93964 and private score 0.93938 and ranked 14th in private.

### validation strategy

A random 0.8/0.2 split are used locally to evaluate the model. We find that for vector space text features,
the difference between local score and leaderboard score can be quite large. Using only text feature and logistic regression, a local score 0.7936 is achieved while public leaderboard score is 0.64333.

At the latter stage, we just fit on the leaderboard.

## features

### simple features

* equality feature: 'title', 'description', 'price', 'categoryID', 'locationID', 'metroID', ('lat', 'lon')
* distance between locations: use ((a['lat'] - b['lat']) \*\* 2 + (a['lon'] - b['lon']) \*\* 2) \*\* (1/2)
* numeric comparison feature: for two values a, b, produce min(a, b), max(a, b), abs(a - b), abs(a - b) / (a + b) if a + b > 0 else 0.0; this comparison is used for char length of title, char length of description, number of images
* jaccard distance for attrsJSON keys and (key, item) pairs.

### simple text comparison

For title and description, we compute following features:
* tokenize text using split by white space or continuous alphanumeric finding, produce jaccard of ngram of two word sets. We use 1gram for title, 1-4 gram for description
* jaro winkler distance
* edit distance ratio: edit_distance(sa, sb) / (len(sa) + len(sb))
* [normalized compression distance](https://en.wikipedia.org/wiki/Normalized_compression_distance) using bz2 and [snappy](https://github.com/google/snappy)

### surrogate features

For each item, when some feature is category, say 'categoryID', 'title', 'locationID', we surrogate it with some numeric values; and add these values to pair features.

These numeric surrogate are aggregation values in item_info(train+test) when group by this feature. We use
* normalized frequent
* mean, std, median of price

### dummy features for categoryID

All pairs has same categoryID. Some dummy features are created for this categoryID:
* one hot encoding
* replace each categoryID with some random integer. we create 20 such permutations

These features give 0.00029 improvement in private leaderboard and 0.00022 in public.

### image comparison features

Generally for each two image, we can get a similarity score. We produce feature by following

1. for item with less images, for each image within, find the most similar image in cosine similarity in another item, record this similarity
2. use the mean, min, max of the similarities in previous step

Three similarity comparison methods are used
* histogram comparison: cv2.calcHist with bin size 8, 32, 64, 128 is used
* image hash: ahash, phash, dhash, whash in [imagehash](https://github.com/JohannesBuchner/imagehash) are used; the similarity is hamming distance
* word2vec: We use pretrained [inception-bn](https://github.com/dmlc/mxnet-model-gallery) to get feature(layer before softmax) for each image. Cosine similarity is used.

Additionally
* for imagehash, jaccard between image arrays based on hash are used
* for word2vec, the similarity between average feature of each item is used

### vector space based text features

There are 110 features in this category based on title and description. The exact process is a bit complicated and messy . Since the total improvement of these features is not big enough, we only give a rough description about there features here. Please refer to feature.corpus_based.py for detail, where we use a make like system to manage data dependency.

The rough process is following
1. tokenzie text: with or without stemming
2. generate bag of word Representation of title and description; 2, 3 gram are generate for title
3. use LSI and nmf to represent text in dense VectorSimilarityFeatureBase
4. pretrained word2vec model from [RusVectōrēs](http://ling.go.mail.ru/dsm/en/about) are used to transform word vector to word2vec space
5. cosine similarity of bag of word and LSI, nmf, word2vec space are used as feature
6. for two tokenized text set A, B, the max idf of word in A\B and B\A are used as feature
7. we use one hot word matrix to predict item price, the predicted price are used as feature
8. for each sentence in description we create feature like in **image comparison feature** using cosine similarity
