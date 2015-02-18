# march-madness-predictions

## 2015 Results
- missed 14 vs. 3 upsets badly
- missed multiple Michigan St. wins
- predicted a confidence of .5 for 4 matchups so not sure who I predicted to win
- correct on 44 of first 62 games (not counting the "ties") correctly, picked Duke to beat Wisconsin in final with confidence=.63
- did slightly worse than seed-based benchmark in Kaggle competition

## TODO

- in-memory DB instead of all the dictionaries of dictionaries
- different classifiers will work better with different features, probably better to pick one instead of the ensemble
- try using different feature selection techniques instead of (or in addition to?) PCA
- make sure new features (and existing?) account for the number of games played in regular season if appropriate
- in 2015 all the games I had confidence=.5 were won by higher seed, that should probably be tie breaker
- what do you do when a team loses early but would have been predicted to beat teams it would face in later rounds, or is predicted to win several games in a row with low confidences?
- checkout http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html for new features
- checkout CP-coding instead of one-hot encoding
- http://googleresearch.blogspot.com/2015/08/the-reusable-holdout-preserving.html
