# 6Decoder

6Decoder employs a neural network based on the Transformer Decoder, treating IPv6 addresses as textual sequences and using generative artificial intelligence to generate IPv6 candidate addresses. 

## Runtime environment

* Python 3.9.1
* pytorch 2.5.1
* pytorch-cuda 12.4

## Seed set

[IPv6 Hitlist](https://ipv6hitlist.github.io/) provides an IPv6 Hitlist Service to publish responsive IPv6 addresses, aliased prefixes, and non-aliased prefixes.

## Run example

1. Expand the compressed IPv6 representation into 32 hexadecimal characters without colons.
```shell
python seed_convert.py --file=data/Seed_S1_10K.txt
```

2. Run 6Decoder experiment program using default hyperparameters. 
```shell
python Run6Decoder.py
```
