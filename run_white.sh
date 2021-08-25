python -m benchmark.FGSM_white --distance=l2 --goal=dodging --model=MobileFace --eps=16 --dataset lfw
python -m benchmark.BIM_white --distance=l2 --goal=dodging --model=MobileFace --eps=16 --dataset lfw
python -m benchmark.MIM_white --distance=l2 --goal=dodging --model=MobileFace --eps=16 --dataset lfw
python -m benchmark.CW_white --goal=dodging --model=MobileFace --eps=16 --dataset lfw


