**General Multi-label Image Classification with Transformers**<br/>
Jack Lanchantin, Tianlu Wang, Vicente Ordóñez Román, Yanjun Qi<br/>
Conference on Computer Vision and Pattern Recognition (CVPR) 2021<br/>
[[paper]](https://arxiv.org/abs/2011.14027) [[poster]](https://github.com/QData/C-Tran/blob/main/supplemental/ctran_poster.pdf) [[slides]](https://github.com/QData/C-Tran/blob/main/supplemental/ctran_slides.pdf)
<br/>



## Training and Running C-Tran ##

Python version 3.7 is required and all major packages used and their versions are listed in `requirements.txt`.

### C-Tran on COCO80 Dataset ###
Download COCO data (19G)
```
wget http://cs.virginia.edu/~jjl5sw/data/vision/coco.tar.gz
mkdir -p data/
tar -xvf coco.tar.gz -C data/
```

Train New Model
```
python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'coco' --use_lmt --dataroot data/
```


### C-Tran on VOC20 Dataset ###
Download VOC2007 data (1.7G)
```
wget http://cs.virginia.edu/~jjl5sw/data/vision/voc.tar.gz
mkdir -p data/
tar -xvf voc.tar.gz -C data/
```

Train New Model
```
python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'voc' --use_lmt --grad_ac_step 2 --dataroot data/
```


## Citing ##

```bibtex
@article{lanchantin2020general,
  title={General Multi-label Image Classification with Transformers},
  author={Lanchantin, Jack and Wang, Tianlu and Ordonez, Vicente and Qi, Yanjun},
  journal={arXiv preprint arXiv:2011.14027},
  year={2020}
}
```
