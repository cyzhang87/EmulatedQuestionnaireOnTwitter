# Understanding Concerns, Sentiments and Disparities of Population Groups During the COVID-19 Pandemic

## Requirements
Ubuntu 16.04.1, python 3.7.5
```
pandas = 1.0.3
numpy = 1.18.2
torch = 1.2.2
matplotlib = 3.2.1
```
## Twitter data
The Twitter data used in this study are collected by Sampled stream API [v1](https://developer.twitter.com/en/docs/labs/sampled-stream/overview) and [v2](https://developer.twitter.com/en/docs/twitter-api/tweets/sampled-stream/introduction) in Twitter developer Labs, which can stream about 1% of publicly available tweets in real-time. Meanwhile, detailed author data of all the tweets are collected to extract population characteristics. Up to November 2020, we have totally collected more than 600 million tweets (over 2 Terabytes) during the COVID-19 pandemic.

## Methods
Please refer to this paper.
Zhang C, Xu S, Li Z, Hu S. Understanding Concerns, Sentiments, and Disparities Among Population Groups During the COVID-19 Pandemic Via Twitter Data Mining: Large-scale Cross-sectional Study. J Med Internet Res. 2021 Mar 5;23(3):e26482. doi: 10.2196/26482. PMID: 33617460; PMCID: PMC7939057.
