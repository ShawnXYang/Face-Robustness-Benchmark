#CUDA_VISIBLE_DEVICES=2 python eval/lfw_test/test_normal.py --model IR50-PGDSoftmax
#CUDA_VISIBLE_DEVICES=2 python eval/lfw_test/test_normal.py --model IR50-PGDArcFace
#python eval/lfw_test/test_normal.py --model IR50-PGDCosFace

#for dataset in "lfw" "ytf"
#do
##    for arch in "IR50-PGDSoftmax" "IR50-PGDArcFace" "IR50-PGDCosFace"
#    for arch in "IR50-PGDAm" "IR50-PGDSphereFace"
#    do
#        echo python eval/$dataset\_test/test_normal.py --model $arch --log log-test/log-$dataset-$arch.txt
#        python eval/$dataset\_test/test_normal.py --model $arch --log log-test/log-$dataset-$arch.txt
#    done
#done

#for arch in "IR50-PGDSoftmax" "IR50-PGDArcFace" "IR50-PGDCosFace" "IR50-PGDAm" "IR50-PGDSphereFace" "CASIA-Softmax" "CASIA-CosFace" "CASIA-ArcFace" "CASIA-SphereFace" "CASIA-Am"
#for arch in "IR50-PGDArcFace" "CASIA-ArcFace" 
#for arch in "IR50-Softmax-BR" "IR50-Softmax-RP" "IR50-Softmax-JPEG"
#for arch in "IR50-Softmax-RP"
for arch in "IR50-TradesCosFace"
do
    python eval/lfw_test/test_normal.py --model $arch --log log-test/log-lfw-$arch.txt
done
