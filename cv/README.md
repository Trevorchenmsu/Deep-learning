
******************************** Dependencies ********************************

channels:
  - pytorch
  - anaconda
  - defaults
dependencies:
  - h5py=3.5.0
  - numpy=1.21.2
  - python=3.8.8
  - scipy=1.7.1
  - setuptools=58.0.4
  - tensorflow-gpu=2.5.0
  - cython=0.29.24
  - pytorch=1.10
  - torchvision=0.11.1
  - pip:
    - scikit-learn==1.0.1
    - opencv-python==4.5.4.58
    - skorch==0.11.0
    - torchsummary


******************************** Input and output ********************************

To get the relavant input and output files and trained models, please visit GT box:
https://gatech.box.com/s/i0hg9ujv3cf8myaslyqleumxefov4648

--------- Inputs ---------
(1) Input dataset:
Stanford Street View House Numbers (SVHN) Dataset: http://ufldl.stanford.edu/housenumbers/
Cropped version is used, which is included in the GT Box (test_32x32.mat and train_32x32.mat)

(2) Input test images:
The images are named as '1.png', '2.png', '3.png', '4.png', '5.png'.
images sources:
https://twitter.com/svpino/status/1298468660990935040
http://www.home-designing.com/buy-modern-and-traditional-house-numbers-for-sale-online
https://www.ebay.com/itm/222489134251
https://www.housesignsdirect.co.uk/illuminated-modern-house-number-sign-40cm-x-24cm.html
https://www.americasfinestlighting.com/

(3) Input video:
A video taken by me is named as 'test_video.mp4'.


--------- Outputs ---------
(1) Trained models
There are four trained models. They are: 
1) model from customized architecture; 
2) model from vgg and trained from scractch;
3) model from vgg with pre-trained weights and trained again (50 epochs);
4) model from vgg with pre-trained weights and trained again (200 epochs).

(2) Output images
There are five output images which are generated from test images after inference.
They are named as 'img_out_1.png', 'img_out_2.png', 'img_out_3.png', 'img_out_4.png', 'img_out_5.png'.

(3) Output videos
There are three output videos that are generated from test_video.mp4 after inference. 
They are named as 'clf_video_customized.mp4', 'clf_video_vgg.mp4', 'clf_video_vgg_pretrained_50.mp4'.

(4) Output statistical images
There are 9 + 2 statistical images. For each model (customized, vgg, vgg_pretrained_50), it has 3 statistical images which includes train-val_loss, train-val_accuracy, and test_accuracy. The remaining 2 statistical images are from vgg_pretrained_200, which is for your information. 

(5) Output data
An Excel file includes the testing results for three models, which is named as predict_results.xlsx


******************************** Model Running Instruction ********************************
Note: put all the code files in the same folder.

If you want to train (mostly not) or test the model (on SVHN testing dataset), you can use the following sample command in the terminal. Make sure you install all the depencies listed above:

<python pipeline.py --model=customized --mode=train --data_dir=./ --model_dir=./>

(1) model: the model to train or test. it can be 'customized', 'vgg', 'vgg_pretrained'

(2) mode: the dataset to select. If 'train' is used, it will read the training dataset for model training. If 'test' is used, it will read the testing dataset for model testing. 

(3) data_dir: the directory where you store the traning or testing dataset.

(4) model_dir: the directory where you store the targeted trained model


If you want to run the model to generate the graded_images based on the provided test images, you can run the command in the terminal:

for images:
<python run.py --model=vgg --data_dir=./test_image --model_dir=./ --result=image --out_dir=./>

for video:
<python run.py --model=vgg --data_dir=./test_video.mp4 --model_dir=./ --result=video --out_dir=./>


(1) model:  the trained model. You can select from 'customized', 'vgg', 'vgg_pretrained'. 
Note: 'vgg' works best, 'customized' is not very stable but the acc is still high. Please don't use 'vgg_pretrained' although the option is provided here because it works worst. 

(2) data_dir: the directory where you store the test images or video.

(3) model_dir: the directory where you store the trained model

(4) result: the output type. It can be image or video.

(5) out_dir: the output directory to store the graded_images or video. 









