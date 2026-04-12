Simple autoencoder trained on ImageNet10k.

```
Running evaluation by patches...

Loading basic from checkpoints/basic_imagenet10k-basic-best.ckpt...
Evaluating model over the validation set...

==============================
Total batches: 112
MSE:  0.005594
PSNR: 22.52 dB
SSIM: 0.6344
==============================

Image saved to outputs/checkpoints_basic_imagenet10k-basic-best.ckpt_comparison.png
Running evaluation of compression...

Loading basic from checkpoints/basic_imagenet10k-basic-best.ckpt...
=============================================
i       Image size      Size before     Size after      ratio   jpeg_ratio
=============================================
0       358x500 313 KB  93 KB              3.34x                     3.7x
1       500x335 224 KB  91 KB              2.46x                     3.4x
2       500x375 310 KB  92 KB              3.37x                     4.7x
3       500x375 383 KB  93 KB              4.10x                     3.6x

Our comparison metrics:

==============================
Total batches: 4
MSE:  0.004369
PSNR: 23.61 dB
SSIM: 0.6777
==============================


JPEG comparison metrics:

==============================
Total batches: 4
MSE:  0.000230
PSNR: 36.29 dB
SSIM: 0.9620
==============================
```