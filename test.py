"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity 
from scipy.stats import pearsonr
import numpy as np
import cv2
import datetime

def calculate_psnr(img1, img2):
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr

def calculate_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = structural_similarity(img1, img2, full=True)
    return ssim_value

def calculate_pearsonr(img1, img2):
    image1_flat = img1.flatten()
    image2_flat = img2.flatten()
    corr, _ = pearsonr(image1_flat, image2_flat)
    return corr

def evaluate_model(model, test_dataset, opt):
    model.eval()
    psnr_list = []
    ssim_list = []
    pear_list = []
    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            real_B = visuals['real_B'].cpu().numpy()  
            fake_B = visuals['fake_B'].cpu().numpy()  

            batch_size = real_B.shape[0]
            for b in range(batch_size):
                real_image = real_B[b].transpose(1, 2, 0) * 255.0
                fake_image = fake_B[b].transpose(1, 2, 0) * 255.0

                real_image = real_image.astype(np.uint8)
                fake_image = fake_image.astype(np.uint8)

                psnr_value = calculate_psnr(real_image, fake_image)
                ssim_value = calculate_ssim(real_image, fake_image)
                pear_value = calculate_pearsonr(real_image, fake_image)

                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                pear_list.append(pear_value)


    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_pear = np.mean(pear_list)
    return avg_psnr, avg_ssim, avg_pear

if __name__ == '__main__':
    opt = TestOptions().parse()  
    opt.num_threads = 0   
    opt.batch_size = 1   
    opt.serial_batches = True  
    opt.no_flip = True    
    opt.display_id = -1   
    dataset = create_dataset(opt)  
    print("testing\n", len(dataset))
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))




    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  
            break
        model.set_input(data)  
        model.test()           
        visuals = model.get_current_visuals()  
        img_path = model.get_image_paths()     
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML

