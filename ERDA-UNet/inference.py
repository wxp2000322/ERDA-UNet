import torch
from model import get_segmentation_model
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import cv2
import os
import time

def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Inference of ERDA')

    #
    # Checkpoint parameters
    #
    parser.add_argument('--pkl-path', type=str, default=r'./DEA_ir_mIoU-0.6069_fmeasure-0.7554.pkl',
                        help='checkpoint path')
    parser.add_argument('--net-name', type=str, default='ERDA',
                        help='net name: R50-ViT-B_16')
    #
    # Test image parameters
    #0
    parser.add_argument('--image-path', type=str, default=r'./XDU106.png', help='image path')
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')

    args = parser.parse_args()
    return args


def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img).unsqueeze(0)

    return preprocessed_img.to(device)


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load network
    print('...load checkpoint: %s' % args.pkl_path)
    # config_vit = CONFIGS_ViT_seg[args.net_name]
    # config_vit.n_classes = 1
    # config_vit.n_skip = 3
    # config_vit.patches.grid = (
    #     16, 16)
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net = get_segmentation_model(args.net_name)
    ckpt = torch.load(args.pkl_path, map_location=torch.device('cpu'))
    net.load_state_dict(ckpt)
    net.to(device)
    net.eval()

    # load image
    print('...loading test image: %s' % args.image_path)
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (args.base_size, args.base_size))) / 255
    input = preprocess_image(img).to(device)  # Make sure the input tensor is on the same device

    # warm up
    for _ in range(5):
        data = torch.ones((2, 3, args.base_size, args.base_size)).to(device)
        net(data)

    # test FPS for 100 inferences
    print('...inference in progress')

    num_iterations = 100  # Number of iterations to measure FPS accurately
    total_time = 0

    for _ in range(num_iterations):
        t1 = time.time()
        with torch.no_grad():
            output = net(input)  # Inference
        t2 = time.time()
        total_time += (t2 - t1)

    avg_time_per_inference = total_time / num_iterations
    fps = 1/ avg_time_per_inference
    print(f"Average FPS over {num_iterations} iterations: {fps:.3f}")

    # start_time = time.time()
    #
    # # Perform 100 inference iterations
    # for _ in range(100):
    #     with torch.no_grad():
    #         output = net(input)
    #
    # end_time = time.time()
    #
    # # Calculate FPS
    # total_time = end_time - start_time
    # fps = 100 / total_time  # FPS is number of inferences divided by total time
    # print("FPS for 100 inferences: ", round(fps, 3))

    # Process output (e.g., thresholding)
    output = output.cpu().detach().numpy().reshape(args.base_size, args.base_size)
    output_g=output>0

    # plt.figure(figsize=(6, 6))
    # plt.imshow(output_g, cmap='gray')
    # plt.axis('off')  # Disable axis to remove white borders
    # plt.savefig(f'{os.path.splitext(os.path.basename(args.image_path))[0]}_DEA_hb.jpg', bbox_inches='tight', pad_inches=0)
    #
    #
    # # Display as heatmap with higher contrast (using 'coolwarm' colormap)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(output, cmap='coolwarm')  # Use 'coolwarm' for high contrast between low (blue) and high (red) values
    # plt.axis('off')  # Disable axis to remove white borders
    # plt.savefig(f'{os.path.splitext(os.path.basename(args.image_path))[0]}_DEA_cs.jpg', bbox_inches='tight', pad_inches=0)

    min_output = np.min(output)
    max_output = np.max(output)

    # 将 output 数据按比例映射到 [-255, 255] 范围内
    scaled_output = ((output - min_output) / (max_output - min_output)) * 510 - 255  # 映射到 [-255, 255]

    cbar = plt.colorbar(plt.imshow(scaled_output, cmap='coolwarm'), orientation='horizontal')  # 水平显示 color bar

    # 设置 color bar 的范围
    cbar.set_ticks([-255, 0, 255])
    cbar.set_ticklabels(['-255', '0', '255'])

    # 将 colorbar 的字体平行于轴
    cbar.ax.tick_params(axis='x', labelrotation=0)  # 字体平行于 colorbar 轴

    # Create 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Generate meshgrid for plotting
    x = np.linspace(0, args.base_size - 1, args.base_size)
    y = np.linspace(0, args.base_size - 1, args.base_size)
    X, Y = np.meshgrid(x, y)

    # Plot the 3D surface with the scaled output
    surf = ax.plot_surface(X, Y, scaled_output, cmap='coolwarm', rstride=1, cstride=1, linewidth=0, antialiased=False,
                           vmin=-255, vmax=255)

    # Invert the z-axis to flip the surface vertically
    ax.invert_zaxis()

    # Hide the axis and grid
    ax.set_axis_off()
    ax.grid(False)

    ax.view_init(20, -60)
    # Save the plot
    plt.savefig(f'{os.path.splitext(os.path.basename(args.image_path))[0]}_DEA_3d.png', bbox_inches='tight',
                pad_inches=0, transparent=True)