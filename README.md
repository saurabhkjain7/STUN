# STUN
Code release for our paper "Stochastic Binary Network for Universal Domain Adaptation'' accepted by WACV 2024.

[Paper](https://openaccess.thecvf.com/content/WACV2024/html/Jain_Stochastic_Binary_Network_for_Universal_Domain_Adaptation_WACV_2024_paper.html) $\cdot$ [Video](https://youtu.be/ntgfBoGRT1c?si=tpSh8EMSD_GMYDLH)

## Requirements
Python 3.8.10, Pytorch 1.10.1, Torch Vision 0.11.2. Use the provided requirements.txt file to create virtual environment.

## Data preparation
[Office Dataset](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?pli=1&resourcekey=0-gNMHVtZfRAyO_t2_WrOunA),
[OfficeHome Dataset](http://hemanthdv.org/OfficeHome-Dataset/), [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

Prepare dataset in data directory.
```
./data/amazon/images/ ## Office
./data/Real ## OfficeHome
./data/visda/train ## VisDA synthetic images
./data/visda/validation ## VisDA real images
```

## Training

All training scripts are stored in the script directory.

Ex. Universal Domain Adaptation on OfficeHome.
```
sh scripts/officehome.sh $gpu-id unida
```
Ex. Open Set Domain Adaptation on office.
```
sh scripts/office.sh $gpu-id oda
```
## Reference codes
Part of our codes are taken from the following Github links:

1.OVANET: https://github.com/VisionLearningGroup/OVANet
2.STAR: https://github.com/zhiheLu/STAR_Stochastic_Classifiers_for_UDA

## Reference
This repository is contributed by [Saurabh Kumar Jain](http://www.cse.iitm.ac.in/profile.php?arg=Mjc4MQ==).
If you consider using this code or its derivatives, please consider citing:

```
@InProceedings{Jain_2024_WACV,
    author    = {Jain, Saurabh Kumar and Das, Sukhendu},
    title     = {Stochastic Binary Network for Universal Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {107-116}
}
```