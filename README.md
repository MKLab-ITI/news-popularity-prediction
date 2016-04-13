# news-popularity-prediction
A set of methods that predict the future values of popularity indices for news posts using a variety of features.

This is supplementary code to the SNOW/WWW'16 workshop paper ["Predicting News Popularity by Mining Online Discussions"](http://dl.acm.org/citation.cfm?doid=2872518.2890096)

The presentation slides from SNOW/WWW'16 can be found [here](http://www.slideshare.net/sympapadopoulos/predicting-news-popularity-by-mining-online-discussions).

This study is also described in a non-technical way in this [blog post](http://www.snow-workshop.org/2016/predicting-news-popularity-by-mining-online-discussions/).

Install
-------
### Required packages
- numpy
- scipy
- pandas
- scikit-learn

### Installation
To install for all users on Unix/Linux:

    python3.4 setup.py build
    sudo python3.4 setup.py install
    
Experiments to reproduce the SNOW/WWW'16 paper's results
--------------------------------------------------------
### Datasets
Three datasets were used in the context of the paper:
- RedditNews
- SlashDot
- BarraPunto

We collected the RedditNews dataset for the context of this paper and as such details on the collection can be found there. Please cite the paper if you intend to use it in your own studies. An anonymized version can be found on the GitHub project page, at news-popularity-prediction.news_popularity_prediction.news_post_data.reddit_news.anonymized_discussions
.

The SlashDot and BarraPunto datasets were made available to us by Drs. Vicenc Gomez and Andreas Kaltenbrunner. We include the code to preprocess them as required for this paper's experiments, although the datasets themselves are not included. Please cite this [paper](http://link.springer.com/article/10.1007/s11280-012-0162-8) if you intend to use them in your own studies.

### Run experiments

Just run:

    python3.4 run_all_experiments.py
    
from news_popularity_prediction.entry_points.snow_2016_workshop. You need to open it first and set the input and output folders.

### Make figures

Just run:

    python3.4 make_all_figures.py
    
from news_popularity_prediction.entry_points.snow_2016_workshop. You need to open it first and set the input and output folders.