import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import html
from scipy.stats import pearsonr
from util import util
import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity 
from scipy.stats import pearsonr


import random
import datetime
from options.test_options import TestOptions


# 计算 PSNR
def calculate_psnr(img1, img2):
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr


# 计算 SSIM
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




def calculate_metrics(path_to_folder):
    fake_folder = os.path.join(path_to_folder, "fake_B")
    real_folder = os.path.join(path_to_folder, "real_B")

    fake_files = sorted(os.listdir(fake_folder))
    real_files = sorted(os.listdir(real_folder))

    psnr_list = []
    ssim_list = []
    pearsonr_list = []

    for fake_file, real_file in zip(fake_files, real_files):
        if fake_file.endswith('.png') and real_file.endswith('.png'):
            fake_img = cv2.imread(os.path.join(fake_folder, fake_file))
            real_img = cv2.imread(os.path.join(real_folder, real_file))

            psnr_value = calculate_psnr(real_img, fake_img)
            ssim_value = calculate_ssim(real_img, fake_img)
            pearsonr_value = calculate_pearsonr(real_img, fake_img)

            print(f"File Pair: {fake_file}")
            print(f"PSNR: {psnr_value}")
            print(f"SSIM: {ssim_value}")
            print(f"Pearsonr: {pearsonr_value}")
            print("-" * 40)

            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
            pearsonr_list.append(pearsonr_value)

    PSNR = np.mean(psnr_list)
    SSIM = np.mean(ssim_list)
    PEAR = np.mean(pearsonr_list)
    print("\nAverage Metrics:")
    print(f"Average PSNR: {np.mean(psnr_list)}")
    print(f"Average SSIM: {np.mean(ssim_list)}")
    print(f"Average Pearsonr: {np.mean(pearsonr_list)}")
    return PSNR, SSIM, PEAR

def evaluate_model(model, test_dataset, opt):
    print("wvanums:\n",len(test_dataset))
    model.eval()
    psnr_list = []
    ssim_list = []
    pear_list = []
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    print(web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            if i > 100:
                break
    path = os.path.join(web_dir, 'images')
    PSNR, SSIM, PEAR = calculate_metrics(path)
    
    return PSNR, SSIM, PEAR


if __name__ == '__main__':
    opt = TrainOptions().parse() 
    train_dataset = create_dataset(opt)  

    test_opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    test_opt.num_threads = 0   # test code only supports num_threads = 1
    test_opt.batch_size = 1    # test code only supports batch_size = 1
    test_opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    test_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    test_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    test_opt.results_dir = ''
    test_dataset = create_dataset(test_opt) 

    

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)
    opt.train_dataset_size = train_dataset_size
    print('The number of training images = %d' % train_dataset_size)
    print('The number of test images = %d' % test_dataset_size)

    model = create_model(opt) 

    visualizer = Visualizer(opt)  
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        iter_data_time = time.time()
        iter_start_time = time.time()

        for i, data in enumerate(train_dataset):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, time.time() - epoch_start_time, t_data)


        if epoch % opt.save_epoch_freq == 0:
            print(f"Saving model at the end of epoch {epoch} (total_iters {total_iters})")
            model.save_networks('latest')
            model.save_networks(epoch)
            print(f"Evaluating model performance at epoch {epoch}...")
            PSNR, SSIM, PEAR = evaluate_model(model, test_dataset, test_opt)
            print("PSNR:", PSNR)
            print("SSIM:", SSIM)
            print("PEAR:", PEAR)
