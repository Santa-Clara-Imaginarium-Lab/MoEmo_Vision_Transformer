import os
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPVisionModel
from pytorchvideo.data.encoded_video import EncodedVideo
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    videoID = 'C0172'
    inputPath = 'D:/Tianma/dataset/original/generate_video/'+videoID+'/angry/'
    ID_lists = os.listdir(inputPath)
    npyPath = 'D:/Tianma/dataset/Pre_process/joints/' + videoID + '_f16.npy'
    # npyPath = dataPath + '/test_3d_'+ID+'_output.npy'

    jointsNPY = np.load(npyPath)

    for name in ID_lists:
        ID = name.split('.')[0]

        np.save('D:/Tianma/dataset/Pre_process/joints/' +videoID+'_'+ ID + '_f16.npy', jointsNPY)
        # 'D:\\Tianma\\dataset\\original\\real\\
        dataPath = inputPath + ID + '.MP4'

        savePath = 'D:/Tianma/dataset/Pre_process/video/'  +videoID+'_'+ ID +'/'
        isExist = os.path.exists(savePath)
        if not isExist:
            os.mkdir(savePath)

        cap = cv2.VideoCapture(dataPath)
        success, image = cap.read()
        count = 0
        begin_index = 0
        output = []
        while success:
            if (count-begin_index) % 4 == 0 and count-begin_index>-1:
                # print(image)
                cv2.imwrite(savePath +str(count)+'.jpg', image)  # save frame as JPEG file
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)

                inputs = processor(images=im_pil, return_tensors="pt")
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                output.append(last_hidden_state)
            if count > begin_index+60:
                break
            success, image = cap.read()
            # print('Read a new frame: ', success)
            count += 1

        output = torch.cat(output, dim=0)
        print(output.shape)
        savePath = 'D:/Tianma/dataset/Pre_process/tensor/'  +videoID+'_'+ ID + '.pt'
        torch.save(output.detach(), savePath)











