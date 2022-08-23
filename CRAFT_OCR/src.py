import os
from craft_text_detector import Craft
import glob
import cv2
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = './weights/transformerocr.pth'
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
detector = Predictor(config)
craft = Craft(output_dir=None, crop_type="box", cuda=False)
def OCR(images):
    # create a craft instance
    
    predictions = []
    # apply craft text detection and export detected regions to output directory
    for image in images:
        prediction_result = craft.detect_text(image)
        predictions.append((image,prediction_result))
    return predictions


def readOCR_splitclass(img, OCR_dict):
  #read txt path
    
    coors = OCR_dict['boxes']

 
    digit = []
    text = []

    for coor in coors:
        x1,y1 = coor[0] 
        x2,y2 = coor[1]
        x3,y3 = coor[2]
        x4,y4 = coor[3]
        top_left_x = int(min([x1,x2,x3,x4]))
        top_left_y = int(min([y1,y2,y3,y4]))
        bot_right_x = int(max([x1,x2,x3,x4]))
        bot_right_y = int(max([y1,y2,y3,y4]))

        img_detect = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
        img_detect = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)
        img_detect = Image.fromarray(img_detect)

        s = detector.predict(img_detect)

        if len(s)>=3:
        # example: [38,296,168,296,168,315,38,315]
            center = [(top_left_x+bot_right_x)//2,(top_left_y+bot_right_y)//2]

            if s[0:2].isdigit() == True:
                digit.append((s,center))
            else:
                text.append((s,center))

    return text,digit

if __name__ == "__main__":
    img = cv2.imread("/home/gumiho/project/QuynhonAI/001.png")
    
    output = OCR([img])

    res = readOCR_splitclass(output[0][0],output[0][1])
    print(res)