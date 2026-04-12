DCAL_2018 trained on combined Div2K and ImageNet10k.
```
Running evaluation by patches...

Loading DCAL_2018 from checkpoints/dcal_combined-DCAL_2018-best.ckpt...
Evaluating model over the validation set...

==============================
Total batches: 162
MSE:  0.006892
PSNR: 21.62 dB
SSIM: 0.5806
==============================

Image saved to outputs/checkpoints_dcal_combined-DCAL_2018-best.ckpt_comparison.png
Running evaluation of compression...

Loading DCAL_2018 from checkpoints/dcal_combined-DCAL_2018-best.ckpt...
=============================================
i       Image size      Size before     Size after      ratio   jpeg_ratio
=============================================
0       2040x1356       5678 KB         1458 KB                    3.89x                     4.1x
1       1632x2040       6979 KB         1703 KB                    4.10x                     3.6x
2       2040x1464       4377 KB         1455 KB                    3.01x                     4.3x
3       1356x2040       5191 KB         1457 KB                    3.56x                     3.8x

Our comparison metrics:

==============================
Total batches: 4
MSE:  0.006128
PSNR: 22.09 dB
SSIM: 0.5137
==============================


JPEG comparison metrics:

==============================
Total batches: 4
MSE:  0.000094
PSNR: 40.27 dB
SSIM: 0.9806
==============================

```