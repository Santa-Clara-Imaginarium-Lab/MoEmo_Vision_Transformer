import cv2
import os
import numpy as np
from PIL import Image
# import paddlehub as hub

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
# def CutVideo2Image(video_path, img_path):
#     cap = cv2.VideoCapture(video_path)
#     index = 0
#     while (True):
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.resize(frame, (800,450))
#             cv2.imwrite(img_path + '%d.jpg' % index, frame)
#             # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # imgs.append(img_rgb)
#             index += 1
#         else:
#             break
#     cap.release()
#     print('Video cut finish, all %d frame' % index)
#
#
# def GetHumanSeg(in_path, out_path):
#     # load model
#     module = hub.Module(name="deeplabv3p_xception65_humanseg")
#     # config
#     frame_path = in_path
#     test_img_path = [os.path.join(frame_path, fname) for fname in os.listdir(frame_path)]
#     input_dict = {"image": test_img_path}
#
#     results = module.segmentation(data=input_dict, output_dir=out_path,use_gpu=True)
#
#
# def BlendImg(fore_image, base_image, output_path):
#     """
#     将抠出的人物图像换背景
#     fore_image: 前景图片，抠出的人物图片
#     base_image: 背景图片
#     """
#     # 读入图片
#     base_image = Image.open(base_image).convert('RGB')
#     fore_image = Image.open(fore_image).resize(base_image.size)
#
#     # 图片加权合成
#     scope_map = np.array(fore_image)[:, :, -1] / 255
#     scope_map = scope_map[:, :, np.newaxis]
#     scope_map = np.repeat(scope_map, repeats=3, axis=2)
#     res_image = np.multiply(scope_map, np.array(fore_image)[:, :, :3]) + np.multiply((1 - scope_map),
#                                                                                      np.array(base_image))
#
#     # 保存图片
#     res_image = Image.fromarray(np.uint8(res_image))
#     res_image.save(output_path)
#
#
# def BlendHumanImg(in_path, screen_path, out_path):
#     humanseg_png = [filename for filename in os.listdir(in_path)]
#     for i, img in enumerate(humanseg_png):
#         img_path = os.path.join(in_path + '%d.png' % (i))
#         output_path_img = out_path + '%d.png' % i
#         BlendImg(img_path, screen_path, output_path_img)
#
#
# def init_canvas(width, height, color=(255, 255, 255)):
#     canvas = np.ones((height, width, 3), dtype="uint8")
#     canvas[:] = color
#     return canvas
#
#
# def GetGreenScreen(width, height, out_path):
#     canvas = init_canvas(width, height, color=(0, 255, 0))
#     cv2.imwrite(out_path, canvas)
#
#
# def CombVideo(in_path, out_path, size):
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(out_path, fourcc, 30.0, size)
#     files = os.listdir(in_path)
#
#     for i in range(len(files)):
#         img = cv2.imread(in_path + '%d.png' % i)
#         # cv2.imshow("test", img)
#         # cv2.waitKey(0)
#         # img = cv2.resize(img, (1280,720))
#         out.write(img)  # 保存帧
#     out.release()
def crop_img(image):
    # Get image semiaxes
    img_h_saxis = image.shape[0]//2
    img_w_saxis = image.shape[1]//2

    # Declare crop semiaxis as the maximum pixels available in BOTH directions
    crop_saxis = min((img_h_saxis, img_w_saxis))

    # Declare center of image
    center = (img_h_saxis, img_w_saxis)

    # Select maximum pixels from center in both directions
    cropped_img = image[(center[0]-crop_saxis): (center[0]+ crop_saxis),
                        (center[1]-crop_saxis): (center[1]+ crop_saxis)]

    # You can include here the resize method

    return cropped_img

def generatorVideo(Video_Path,GreenScreen_Path,savePath):
    video = cv2.VideoCapture(Video_Path)
    image = cv2.imread(GreenScreen_Path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    size = (450, 450)

    # Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
    result = cv2.VideoWriter(savePath,
                             cv2.VideoWriter_fourcc(*'X264'),
                             24, size)
    #
    while True:
        ret, frame = video.read()
        if ret == False:
            break
        # img = frame
        # # change to hsv
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = crop_img(frame)
        image = crop_img(image)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        mask1 = cv2.inRange(frame, (35, 90, 166), (90, 235, 255))

        ## final mask and masked
        mask = cv2.bitwise_or(mask1, mask1)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        f = frame - res
        f = np.where(f == 0, image, f)
        #
        # cv2.imshow("video", frame)
        # cv2.imshow("mask", f)
        result.write(cv2.cvtColor(f, cv2.COLOR_HSV2BGR))



    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Config
    # Video_Path = 'video/0.mp4'



    backFolder = 'thinking'
    backgroundPath = 'D:/Tianma/dataset/background/' + backFolder + '/'
    GreenScreen_Path = backgroundPath

    backSubFolder_lists = os.listdir(backgroundPath)
    for backSubFolder in backSubFolder_lists:
        print('class:',backSubFolder)
        GreenScreen_Path = os.path.join(backgroundPath, backSubFolder)

        IMG_lists = os.listdir(GreenScreen_Path)

        # ID = 'C0090'
        # 'D:\\Tianma\\dataset\\original\\real\\
        Video_Path = 'D:/Tianma/dataset/original/green_good/'
        Video_lists = os.listdir(Video_Path)

        for green_video in Video_lists:
            ID = green_video.split('.')[0]
            print('ID :',ID)
            if ID != 'C0172':
                continue
            green_video_path = os.path.join(Video_Path, green_video)
            savePath = 'D:/Tianma/dataset/original/generate_video/' + ID +'/'
            ComOut_Path = savePath
            isExist = os.path.exists(savePath)
            if not isExist:
                os.mkdir(savePath)
            savePath = savePath + 'angry/'
            ComOut_Path = savePath
            isExist = os.path.exists(savePath)
            if not isExist:
                os.mkdir(savePath)

            for IMG_Name in IMG_lists:
                print(IMG_Name)
                IMG_path = os.path.join(GreenScreen_Path, IMG_Name)
                saveVideo = savePath + backFolder + '_' + backSubFolder + '_' + IMG_Name.split('.')[0] + '.mp4'
                generatorVideo(green_video_path, IMG_path, saveVideo)


    # # 第一步：视频->图像
    #
    # # if not os.path.exists(FrameCut_Path):
    # #     os.mkdir(FrameCut_Path)
    # # CutVideo2Image(Video_Path, FrameCut_Path)
    # #
    # # 第二步：抠图
    # if not os.path.exists(FrameSeg_Path):
    #     os.mkdir(FrameSeg_Path)
    # GetHumanSeg(FrameCut_Path, FrameSeg_Path)
    #
    # # 第三步：生成绿幕并合成
    # if not os.path.exists(GreenScreen_Path):
    #     GetGreenScreen(1920, 1080, GreenScreen_Path)
    #
    # if not os.path.exists(FrameCom_Path):
    #     os.mkdir(FrameCom_Path)
    # BlendHumanImg(FrameSeg_Path, GreenScreen_Path, FrameCom_Path)
    #
    # # 第四步：合成视频
    # if not os.path.exists(ComOut_Path):
    #     CombVideo(FrameCom_Path, ComOut_Path, (1920, 1080))