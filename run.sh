rm -rf ./logs/
rm -rf ./save/
rm loss.csv
CUDA_VISIBLE_DEVICES=0 python3 main.py --train_dqn
