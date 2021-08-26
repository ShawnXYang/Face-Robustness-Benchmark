Running Benchmarks
==================

Using Command Line
----------------------

For the white-box evaluation, you can run ``FGSM`` evaluation on ``MobileFace`` for ``LFW`` dataset in command line:

.. code-block:: shell

    python benchmark/lfw/FGSM_white.py --distance=l2 --goal=dodging --model=MobileFace --eps=16 --log=log-lfw-FGSM-l2-dodging-MobileFace-white.txt 

Then you can obtain:

.. code-block:: shell
    
    adv_img,tar_img,score,dist,success
    1.npy,data/lfw-112x112/Abel_Pacheco/Abel_Pacheco_0004.jpg,0.21092090010643005,1.0467989629677874,1
    2.npy,data/lfw-112x112/Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg,0.21074934303760529,4.202811928700617,1
    3.npy,data/lfw-112x112/Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg,0.21039743721485138,2.1047161963395666,1
    4.npy,data/lfw-112x112/Amber_Tamblyn/Amber_Tamblyn_0002.jpg,0.20931993424892426,1.2771732226518993,1
    ....

``score`` indicates the similarity predicted by victim model, ``dist`` means the minimal adversarial or distortion distance, and ``success`` means whether this attack is successful.

For the black-box evaluation, you can run  the following command line:

.. code-block:: shell
    
    # generate adversarial examples
    python benchmark/lfw/FGSM_black.py --distance=l2 --goal=dodging --model=MobileFace --eps=4 --output=outputs/lfw-FGSM-l2-dodging-MobileFace --batch_size=20
    # generate log file
    python benchmark/lfw/run_test.py --model=Mobilenet --distance=l2 --anno=outputs/lfw-FGSM-l2-dodging-MobileFace/annotation.txt --log=log-lfw-Mobilenet-FGSM-l2-dodging-MobileFace-black.txt --goal=dodging 

After executing the first script, the adversarial examples are saved as png files in --output. An annotation file (``annotation.txt``) is also saved in ``--output``.

Then ``run_test.py`` will generate the evaluation log file in ``--log``.
