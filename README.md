
# Guide

Download the datasets first

```bash
./datasets/download.sh
```

## Resources
Original repo: [CycleGAN](https://github.com/junyanz/CycleGAN/)<br/>
Comparing implementations in the following frameworks:
* Tensorflow
* PyTorch: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

### PyTorch
Install visdom in order to visualise the training over time:
```
pip install visdom
python -m server.visdom
```
The server will be accessible at https://localhost:8097

#### Set up model
Enter the PyTorch folder using `cd pytorch`.
Download datasets:
``` ./datasets/download_cyclegan_dataset.sh [dataset_name]```

Edit scripts/train_cyclegan.sh with dataset of choice
```bash
set -ex
python3 train.py --dataroot ./datasets/[dataset_name] --name [name] --model cycle_gan --pool_size 50 --no_dropout
```
If your machine is not equipped with an NVIDIA GPU, you can add `--gpu_id -1` to train on the CPU (although this is not recommended and will take a long time). 
Checkpoints and samples are saved in the checkpoint folder.

Our CycleGAN setup took ~20 hours to train over 200 epochs on the summer2winter_yosemite dataset.

![](https://github.com/bubba/cs4404-project-1/blob/master/pytorch/results/summer2winter_yosemite_0.png "Example 0")
![](https://github.com/bubba/cs4404-project-1/blob/master/pytorch/results/summer2winter_yosemite_1.png "Example 1")
![](https://github.com/bubba/cs4404-project-1/blob/master/pytorch/results/summer2winter_yosemite_2.png "Example 2")
![](https://github.com/bubba/cs4404-project-1/blob/master/pytorch/results/summer2winter_yosemite_3.png "Example 3")

 
