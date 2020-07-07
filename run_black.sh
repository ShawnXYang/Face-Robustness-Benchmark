## LFW
# generate adversarial examples
python benchmark/lfw/FGSM_black.py --distance=linf --goal=dodging --model=MobileFace --eps=8 --output=outputs/lfw-FGSM-linf-dodging-MobileFace --batch_size=20
# generate log file
python benchmark/lfw/run_test.py --model=Mobilenet --distance=linf --anno=outputs/lfw-FGSM-linf-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-FGSM-linf-dodging-MobileFace-black.txt --goal=dodging 

## YTF
# generate adversarial examples
python benchmark/ytf/FGSM_black.py --distance=linf --goal=dodging --model=MobileFace --eps=8 --output=outputs/ytf-FGSM-linf-dodging-MobileFace --batch_size=20 
# generate log file
python benchmark/ytf/run_test.py --model=Mobilenet --distance=linf --anno=outputs/ytf-FGSM-linf-dodging-MobileFace/annotation.txt --log=log-ytf-Mobilenet-FGSM-linf-dodging-MobileFace-black.txt --goal=dodging 


# More examples
#python benchmark/lfw/FGSM_black.py --distance=linf --goal=dodging --model=MobileFace --eps=8 --output=/data/dingcheng/outputs/FGSM-linf-dodging-MobileFace --batch_size=20 
#python benchmark/lfw/BIM_black.py --distance=linf --goal=dodging --model=MobileFace --eps=8 --output=/data/dingcheng/outputs/BIM-linf-dodging-MobileFace --batch_size=20 --iters=100 
#python benchmark/lfw/CIM_black.py --distance=linf --goal=dodging --model=MobileFace --eps=8 --output=/data/dingcheng/outputs/CIM-linf-dodging-MobileFace --batch_size=20 --iters=100 
#python benchmark/lfw/LGC_black.py --distance=linf --goal=dodging --model=MobileFace --eps=8 --output=/data/dingcheng/outputs/LGC-linf-dodging-MobileFace --batch_size=20 --iters=100 
#python benchmark/lfw/MIM_black.py --distance=linf --goal=dodging --model=MobileFace --eps=8 --output=/data/dingcheng/outputs/MIM-linf-dodging-MobileFace --batch_size=20 --iters=100 
#python benchmark/lfw/run_test.py --model=Mobilenet --distance=linf --anno=/data/dingcheng/outputs/FGSM-linf-dodging-MobileFace/annotation.txt --log=log-Mobilenet-FGSM-linf-dodging-MobileFace-black.txt --goal=dodging 
#python benchmark/lfw/run_test.py --model=Mobilenet --distance=linf --anno=/data/dingcheng/outputs/BIM-linf-dodging-MobileFace/annotation.txt --log=log-Mobilenet-BIM-linf-dodging-MobileFace-black.txt --goal=dodging 
#python benchmark/lfw/run_test.py --model=Mobilenet --distance=linf --anno=/data/dingcheng/outputs/CIM-linf-dodging-MobileFace/annotation.txt --log=log-Mobilenet-CIM-linf-dodging-MobileFace-black.txt --goal=dodging 
#python benchmark/lfw/run_test.py --model=Mobilenet --distance=linf --anno=/data/dingcheng/outputs/LGC-linf-dodging-MobileFace/annotation.txt --log=log-Mobilenet-LGC-linf-dodging-MobileFace-black.txt --goal=dodging 
#python benchmark/lfw/run_test.py --model=Mobilenet --distance=linf --anno=/data/dingcheng/outputs/MIM-linf-dodging-MobileFace/annotation.txt --log=log-Mobilenet-MIM-linf-dodging-MobileFace-black.txt --goal=dodging 
