# LFW
python benchmark/lfw/FGSM_white.py --distance=l2 --goal=dodging --model=MobileFace --eps=16 --log=log-lfw-FGSM-l2-dodging-MobileFace-white.txt 
python benchmark/lfw/CW_white.py --goal=dodging --model=MobileFace --eps=16 --log=log-lfw-CW-dodging-MobileFace-white.txt 
python benchmark/lfw/Evolutionary_white.py --distance=l2 --goal=dodging --model=MobileFace --log=log-lfw-Evo-l2-dodging-MobileFace-white.txt  --batch_size 1

# YTF
python benchmark/ytf/FGSM_white.py --distance=l2 --goal=dodging --model=MobileFace --eps=16 --log=log-ytf-FGSM-l2-dodging-MobileFace-white.txt 
