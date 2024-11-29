import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import torch
import os
import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix
from torch import Tensor
from fixedPoint.nn.splitArrayArithmetic import splitArr_matmul_nn
from fixedPoint.nn.commonConst import ConstForSplitArray as cfs, phyArrParams, phyArrMode, debug
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple

def ISC(img: ndarray, knl: ndarray, device) -> ndarray:
    M, N = img.shape
    m, n = knl.shape # assume m == n
    input_kernel_fifo = np.zeros(shape=(M, M+m-1, n), dtype=np.float32)
    for i in range(M):
        input_kernel_fifo[i, i:i+m, :] = knl
    # print(input_kernel_fifo.shape)
    input_kernel_fifo = np.transpose(input_kernel_fifo, (0, 2, 1)) # [M, n, M+m-1]
    print(input_kernel_fifo.shape)
    img_pad = F.pad(torch.Tensor(img).to(device), (n//2, n//2, m//2, m//2), mode='constant', value=0)
    print(img_pad.shape)
    # print(torch.isnan(img_pad).any(), np.isnan(input_kernel_fifo).any())
    output = splitArr_matmul_nn(torch.Tensor(input_kernel_fifo).reshape(-1, M+m-1).to(device), img_pad, cfs.input_bit_width, cfs.dac_bit_width, cfs.adc_bit_width, cfs.output_bit_width, cfs.weight_bit_width, phyArrParams.defaultArrMode, torch.float32)
    output = output.reshape(M, n, N+n-1)
    conv_output = torch.zeros((M, N)).to(device)
    for i in range(n):
        conv_output += output[:, i, i:N+i]
    return conv_output.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='PyTorch Image Processing Example')
    parser.add_argument('--image-dir', default='data/image-processing', metavar='DD',
                        help='dir of image')
    parser.add_argument('--image-path', default='Lena.bmp', metavar='DN',
                        help='image file name')
    parser.add_argument('--operation', type=int, default=0, metavar='DN',
                        help='image operation (0:compression 1:edge_detection -1:none)')
    parser.add_argument('--method', type=int, default=0, metavar='MD',
                        help='use which iteration method (comp: 0:dct/idct -1:none, other: 0:canny -1:none, other: 0:xxx -1:none)')
    parser.add_argument('--compression-size', default=[200, 200], metavar='T',
                        help='')
    parser.add_argument('--half-float', action='store_true', default=False,
                        help='For use 16b float')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA for acceleration')
    parser.add_argument('--cuda_use_num', type=int, default=3, metavar='CUDA',
                        help='use which cuda (choice: 0-2)')
    args = parser.parse_args()
    print(args)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_use_num) if use_cuda else "cpu")
    print(f"device: {device}")

    image = cv2.imread(os.path.join(args.image_dir, args.image_path), 0) # 目前只支持灰度图
    image = image.astype(np.float32)
    # image = np.random.randint(0, 256, size=(20, 30)).astype(np.float32)
    image_size = image.shape
    print(image_size)

    if args.operation == 0:
        # image compression
        if args.method == 0:
            ### discrete cosine transform
            # image: f(x,y), size m*n
            # A: size m*m
            # B: size n*n
            # dct image: F(x,y)=Af(x,y)B'
            # idct image: f'(x,y)=A'F(x,y)B
            x = u = np.arange(image_size[0], dtype=np.float32)
            y = v = np.arange(image_size[1], dtype=np.float32)
            A = np.cos(((x*2+1)*u.reshape(-1, 1)*np.pi)/(2*image_size[0]))*np.sqrt(2/image_size[0])
            A[0, :] /= np.sqrt(2)
            # print(A)
            B = np.cos(((y*2+1)*v.reshape(-1, 1)*np.pi)/(2*image_size[1]))*np.sqrt(2/image_size[1])
            B[0, :] /= np.sqrt(2)
            # print(np.transpose(B))

            ### software builtin dct
            img_dct = cv2.dct(image)
            half = np.sort(np.abs(img_dct), axis=None)[image_size[0]*image_size[1]//2]
            img_dct[np.abs(img_dct) < half] = 0
            img_idct = cv2.idct(img_dct)

            ### software dct
            image_dct = np.matmul(np.matmul(A, image), np.transpose(B))
            half = np.sort(np.abs(image_dct), axis=None)[image_size[0]*image_size[1]//2]
            image_dct[np.abs(image_dct) < half] = 0
            image_idct = np.matmul(np.matmul(np.transpose(A), image_dct), B)

            ### hardware dct
            temp1 = splitArr_matmul_nn(torch.Tensor(np.transpose(image)).to(device), torch.Tensor(np.transpose(A)).to(device), cfs.input_bit_width, cfs.dac_bit_width, cfs.adc_bit_width, cfs.output_bit_width, cfs.weight_bit_width, phyArrParams.defaultArrMode, torch.float32)
            hw_dct = splitArr_matmul_nn(torch.transpose(temp1, 0, 1), torch.Tensor(np.transpose(B)).to(device), cfs.input_bit_width, cfs.dac_bit_width, cfs.adc_bit_width, cfs.output_bit_width, cfs.weight_bit_width, phyArrParams.defaultArrMode, torch.float32)
            # compression
            hw_dct_comp = hw_dct.clone()
            half = hw_dct.abs().flatten().sort()[0][image_size[0]*image_size[1]//2]
            hw_dct_comp[hw_dct_comp.abs() < half] = 0
            # hardware idct
            temp2 = splitArr_matmul_nn(torch.transpose(hw_dct_comp, 0, 1), torch.Tensor(A).to(device), cfs.input_bit_width, cfs.dac_bit_width, cfs.adc_bit_width, cfs.output_bit_width, cfs.weight_bit_width, phyArrParams.defaultArrMode, torch.float32)
            hw_idct = splitArr_matmul_nn(torch.transpose(temp2, 0, 1), torch.Tensor(B).to(device), cfs.input_bit_width, cfs.dac_bit_width, cfs.adc_bit_width, cfs.output_bit_width, cfs.weight_bit_width, phyArrParams.defaultArrMode, torch.float32)
            
            hw_idct = hw_idct.cpu().numpy()
            print("hw max min value = {}, {}".format(hw_idct.max(), hw_idct.min()))
            
            # hw_idct[hw_idct<0] = 0
            # hw_idct[hw_idct>255] = 255
            soft_min, soft_max = image.min(), image.max()
            print(f"softwaree max, min = {soft_max}, {soft_min}")
            hd_min, hd_max = hw_idct.min(), hw_idct.max()

            hw_idct_recover = (hw_idct-hd_min)/(hd_max-hd_min)*(soft_max-soft_min) + soft_min
            hw_idct_recover[hw_idct_recover>255] = 255
            hw_idct_recover[hw_idct_recover<0] = 0

            print("after max min recalculate, hw max min value = {}, {}".format(hw_idct_recover.max(), hw_idct_recover.min()))
            f_max = 255  # f_max is 2^8 - 1
            MSE = np.square(hw_idct-image).sum()/(image_size[0]*image_size[1])
            PSNR = 10*np.log10(f_max**2/MSE)
            MSE_recover = np.square(hw_idct_recover-image).sum()/(image_size[0]*image_size[1])
            PSNR_recover = 10*np.log10(f_max**2/MSE_recover)
            print(f"MSE = {MSE}, PSNR = {PSNR}, MSE_recover = {MSE_recover}, PSNR_recover = {PSNR_recover}")

            plt.figure()
            im = plt.imshow(image_idct, cmap='Greys')
            # plt.savefig("./image.jpg")
            cv2.imwrite("image_out.bmp", hw_idct)
            cv2.imwrite("image_recover_out.bmp", hw_idct_recover)

            plt.figure()
            im = plt.imshow(hw_idct, cmap='Greys')
            # plt.savefig("./image2.jpg")
    elif args.operation ==  1:
        # edge detection
        if args.method == 0:
            # canny edge detection algo
            # 1. 灰度化（选择灰度图或者使用python builtin库）
            # 2. 高斯滤波
            # 3. 计算梯度幅值与方向
            # 4. 非极大值抑制
            # 5. 双阈值算法检测和连接边缘
            TL = 0.1
            TH = 0.2
            Gauss_kernel_3x3 = 1/16*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
            Gauss_kernel_5x5 = 1/273*np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]], dtype=np.float32)
            Sobel_x = np.array([[-2, 0, 2], [-2, 0, 2], [-2, 0, 2]], dtype=np.float32)
            Sobel_y = np.array([[2, 2, 2], [0, 0, 0], [-2, -2, -2]], dtype=np.float32)
            # 2. 高斯滤波
            conv1_img = ISC(image, Gauss_kernel_3x3, device=device)
            # 3. 计算梯度幅值与方向
            conv2_img = ISC(conv1_img, Sobel_x, device=device)
            conv3_img = ISC(conv2_img, Sobel_y, device=device)
            G_img = np.sqrt(np.square(conv2_img)+np.square(conv3_img))
            theta = np.arctan(conv3_img/conv2_img)
            # 4. 非极大值抑制
            # 5. 双阈值算法检测和连接边缘
            G_img = (G_img-G_img.min())/(G_img.max()-G_img.min())
            G_process = np.copy(G_img)
            G_process[G_process>TH] = 1
            G_process[G_process<TL] = 0
            G_process_pad = F.pad(torch.Tensor(G_process), (1, 1, 1, 1), mode='constant', value=0).numpy()
            M, N = image.shape
            for i in range(1, M+1):
                for j in range(1, N+1):
                    if G_process_pad[i, j] > TL and G_process_pad[i, j] < TH:
                        if G_process_pad[i-1, j-1] > TH or G_process_pad[i-1, j] > TH or G_process_pad[i-1, j+1] > TH or G_process_pad[i, j-1] > TH \
                        or G_process_pad[i, j+1] > TH or G_process_pad[i+1, j-1] > TH or G_process_pad[i+1, j] > TH or G_process_pad[i+1, j+1] > TH:
                            G_process_pad[i, j] = 1
                        else:
                            G_process_pad[i, j] = 0

            G_processed = G_process_pad[1:M+1, 1:N+1]

            print(G_processed.shape)

            plt.figure()
            im = plt.imshow(image, cmap='Greys')
            # im = plt.imshow(image)
            plt.savefig("./image.jpg")

            plt.figure()
            im = plt.imshow(G_processed, cmap='Greys')
            # im = plt.imshow(G_processed)
            plt.savefig("./image2.jpg")


if __name__ == "__main__":
    main()
