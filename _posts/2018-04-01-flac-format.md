---
layout: post
title: The .flac format: linear predictive coding and rice codes
mathjax: true
---

*FLAC* stands for *F*ree *L*ossless *A*udio *C*odec. It's the standard open source codec used nowadays for lossless audio compression and it's the one one powering Tidal and that any audiophile friend of your keeps recommanding. Personally I had no clue how audio lossless compression works and I was surprised to discover how simple the underlyng concepts are. In this post I would like to take the .flac format as a reference example and focus the two core step of the process of lossless audio compression: linear predictive coding and Rice coding.

TL;TR
All lossless audio compression codecs work (sort of) the same: first the audio stream is split into blocks or frames, and then each one is compressed. Compressing means identifying structure in the signal. Any structure is redundant by definition and can be intead represented more efficiently using a mathematical model and its parameters. Usually a very simple model is used, some form of linear predictive coding (for the statisticians: is just a linear autoregressive model). Since the compressed representation needs to be lossless  the residuals are then encoded, but using a coding scheme optimised for their distribution. Linear predictive coding leads to residual Laplace-distributed and hence usually the Rice coding scheme is used. 

# The .flac format

<picture of the sream>
  
The basic structure of a FLAC stream is:
- The four byte string "fLaC"
- The STREAMINFO metadata block
- Zero or more other metadata blocks
- One or more audio frames

FLAC defines various metadata blocks. They can be used for padding, seek tables, tags, cuesheets, and even application-specific data. Funnyly enough there is not a metadata block for the famous ID3 tags (where the artist, album, etc info are usually stored) but the world doesn't care and most decoders, including the reference one, know how to handle them.

The only mandatory block is the STREAMINFO block. This block contains information like the sample rate, number of channels, etc., and data that can help the decoder manage its buffers, like the minimum and maximum data rate and minimum and maximum block size. Also included in the STREAMINFO block is the MD5 signature of the unencoded audio data, useful for checking an entire stream for transmission errors.

Following the metadata blocks there is the sequence of frames containing the compressed audio stream. FLAC splits the unencoded audio data into blocks, and encodes each block separately. The encoded block is then packed into a frame whose header contains the compression parameters, and appended to the stream. Blocks need not to be the same size (although the reference implementation does it). This serves two purposes: it is possible to uncompress, edit and recompress only a subset of the frames at a time and the compression parameters can change over time, which is ideal since the audio signal is most certaintly not stationary. There is a trade-off in choosing the frame size as each frame header comes with a cost. FLAC deafaults to 4096 samples which at a sampling rate of 44.1kHz equals about 93 milliseconds.

# Compressing the audio signal
But how does the compression of the signal happen? 

The raw encoding of an audio signal is extremely inefficient: a 16 bit 44.1kHz recording will save each second as a sequence of 44100 16 bit numbers recording the value of the audio wave over time. Any structure in the audio signal will just be encoded as it comes: any moment of silence takes as much space as the most explosive of the cymbals. Compressing algorithm try to identify and model the structure of the signal so that they can trasmit the model instead, effectively transmitting a big part of the original signal all at once. After the modelling stage they identify which parts of the signal waere *left over* by the overall indentified structure (the residuals) and they encode it so that the signal can be reconstructed exactly.



# Rice coding



