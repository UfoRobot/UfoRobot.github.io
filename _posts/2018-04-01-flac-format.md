---
layout: post
title: The .flac format: linear predictive coding and rice codes
mathjax: true
---

*FLAC* stands for *F*ree *L*ossless *A*udio *C*odec.

# General structure of most lossless audio compression algorithms

The stream is split in *frames* (also called blocks) of equal size. Each frame contains a frame header which itself contains the parameters
of the compressed signal and potentially additional metadata. This serves two purposes: it is possible to uncompress, edit and
recompress only a subset of the frames at a time and the compression parameters can change over time, which is ideal since the audio signal 
is most certaintly not stationary. There is a trade-off in choosing the frame size as each frame header comes with a cost.
flac deafaults to 4096 samples which at a sampling rate of 44.1kHz equals about 93 milliseconds.

The content of each frame is then decorellated / modelled / compressed. At this stage a model is fitted to the signal which is thus
compressed by being replaced by the model itself. Most commmonly the model used is some form of linear predictive coding, which is 
the singnal-prcessing name of an autoregressive model. The residuals of the model are then assumed to be uncorrelated.

The residuals are then encoded using a coding scheme optimised for their distribution. This allows the signal to be reconstructed exactly
with no loss of information.

