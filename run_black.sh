# FGSM
python -m RobFR.benchmark.FGSM_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/lfw-FGSM-l2-dodging-MobileFace --batch_size=40 --dataset lfw
python -m RobFR.benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/lfw-FGSM-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-FGSM-l2-dodging-MobileFace-black.txt --goal=dodging 
python -m RobFR.benchmark.run_score --goal=dodging --log=log-lfw-Mobilenet-FGSM-l2-dodging-MobileFace-black.txt

# BIM
python -m RobFR.benchmark.BIM_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/lfw-BIM-l2-dodging-MobileFace --batch_size=40 --dataset lfw --iters 100
python -m RobFR.benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/lfw-BIM-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-BIM-l2-dodging-MobileFace-black.txt --goal=dodging 
python -m RobFR.benchmark.run_score --goal=dodging --log=log-lfw-Mobilenet-BIM-l2-dodging-MobileFace-black.txt

# MIM
python -m benchmark.MIM_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/lfw-MIM-l2-dodging-MobileFace --batch_size=30 --dataset lfw
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/lfw-MIM-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-MIM-l2-dodging-MobileFace-black.txt --goal=dodging 
python -m benchmark.run_score --goal=dodging --log=log-lfw-Mobilenet-MIM-l2-dodging-MobileFace-black.txt

# CIM
python -m benchmark.CIM_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/lfw-CIM-l2-dodging-MobileFace --batch_size=30 --dataset lfw
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/lfw-CIM-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-CIM-l2-dodging-MobileFace-black.txt --goal=dodging 
python -m benchmark.run_score --goal=dodging --log=log-lfw-Mobilenet-CIM-l2-dodging-MobileFace-black.txt

# LGC
python -m benchmark.LGC_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/lfw-LGC-l2-dodging-MobileFace --batch_size=30 --dataset lfw
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/lfw-LGC-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-LGC-l2-dodging-MobileFace-black.txt --goal=dodging 
python -m benchmark.run_score --goal=dodging --log=log-lfw-Mobilenet-LGC-l2-dodging-MobileFace-black.txt

# DIM
python -m benchmark.DIM_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/lfw-DIM-l2-dodging-MobileFace --batch_size=30 --dataset lfw --use_lgc
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/lfw-DIM-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-DIM-l2-dodging-MobileFace-black.txt --goal=dodging 
python -m benchmark.run_score --goal=dodging --log=log-lfw-Mobilenet-DIM-l2-dodging-MobileFace-black.txt

# TIM
python -m benchmark.TIM_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/lfw-TIM-l2-dodging-MobileFace --batch_size=30 --dataset lfw --use_lgc
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/lfw-TIM-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-TIM-l2-dodging-MobileFace-black.txt --goal=dodging 
python -m benchmark.run_score --goal=dodging --log=log-lfw-Mobilenet-TIM-l2-dodging-MobileFace-black.txt

# LGC-YTF
python -m benchmark.LGC_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/ytf-LGC-l2-dodging-MobileFace --batch_size=30 --dataset ytf
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/ytf-LGC-l2-dodging-MobileFace/annotation.txt --log=log-ytf-Mobilenet-LGC-l2-dodging-MobileFace-black.txt --goal=dodging --dataset ytf
python -m benchmark.run_score --goal=dodging --log=log-ytf-Mobilenet-LGC-l2-dodging-MobileFace-black.txt

# BIM-YTF
python -m benchmark.BIM_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/ytf-BIM-l2-dodging-MobileFace --batch_size=30 --dataset ytf
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/ytf-BIM-l2-dodging-MobileFace/annotation.txt --log=log-ytf-Mobilenet-BIM-l2-dodging-MobileFace-black.txt --goal=dodging --dataset ytf
python -m benchmark.run_score --goal=dodging --log=log-ytf-Mobilenet-BIM-l2-dodging-MobileFace-black.txt

# LGC-CFP
python -m benchmark.LGC_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/cfp-LGC-l2-dodging-MobileFace --batch_size=30 --dataset cfp
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/cfp-LGC-l2-dodging-MobileFace/annotation.txt --log=log-cfp-Mobilenet-LGC-l2-dodging-MobileFace-black.txt --goal=dodging --dataset cfp
python -m benchmark.run_score --goal=dodging --log=log-cfp-Mobilenet-LGC-l2-dodging-MobileFace-black.txt

# BIM-CFP
python -m benchmark.BIM_black --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=/data1/yangdc/face_attack_outputs/cfp-BIM-l2-dodging-MobileFace --batch_size=30 --dataset cfp
python -m benchmark.run_test --model=Mobilenet --distance=l2 --anno=/data1/yangdc/face_attack_outputs/cfp-BIM-l2-dodging-MobileFace/annotation.txt --log=log-cfp-Mobilenet-BIM-l2-dodging-MobileFace-black.txt --goal=dodging --dataset cfp
python -m benchmark.run_score --goal=dodging --log=log-cfp-Mobilenet-BIM-l2-dodging-MobileFace-black.txt