set -ex
python3 train.py --gpu_id -1 --dataroot ./datasets/summer2winter_yosemite --name summer2winter_cyclegan --model cycle_gan --pool_size 50 --no_dropout
