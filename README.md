# march-madness-predictions

Predict win probabilities for tournement games in Kaggle March Machine Learning Mania. Generate bracket based on probabilities to be used in traditional pools. Learn about machine learning.

## 2015 Results
- missed 14 vs. 3 upsets badly
- missed multiple Michigan St. wins and early Villanova loss
- correct on 44 of first 62 games (not counting the "ties"), picked Duke to beat Wisconsin in final with confidence=.63
- did slightly worse than [seed-based benchmark][] in Kaggle competition

## 2016 Results

## TODO

- use virtualenv and a requirements.txt for dependencies
- add headers to transformation data
- results vary every run, find out how to make it more stable
- use sklearn pipeline (implement blending step) to include feature selection in CV

## FUTURE IDEAS

- develop other rankings based on Colley or Massey methods (see http://netprophetblog.blogspot.com/2015/09/massey-example.html and https://www.kaggle.com/c/march-machine-learning-mania-2016/forums/t/19551/converting-spreads-to-win-percentages-and-vice-versa)
- add more ML techniques
- visualizations


[seed-based benchmark]: https://www.kaggle.com/c/march-machine-learning-mania-2016/forums/t/18902/understanding-the-benchmark-submissions