import json
from flask import Flask, request, json, render_template
import base64
from Segment.test_segment import Segment
import cv2
from CRAFT_OCR.src import OCR, readOCR_splitclass
from Transformer.main_transformer import transfer_English
import glob
import re
import numpy as np
import time
app = Flask(__name__)

# TODO
# Import model

def convert_price(prices):
    for i in range(len(prices)):
        new_prices = prices[i].replace("k","000")
        if prices[i]!="NOT GIVEN":
            new_prices = re.sub("\D","",new_prices)
        if len(new_prices)<4:
            new_prices +="000"
        prices[i] = new_prices
    return prices

def match_price(dishes,prices):
    Vietnamese_dish = []
    price_list = []
    for dish in dishes:
        if len(prices)==0:
            Vietnamese_dish.append(dish[0])
            price_list.append("NOT GIVEN")
            continue
        distance = [(dish[1][0]-price[1][0])**2+(dish[1][1]-price[1][1])**2 for price in prices]
        inde = distance.index(min(distance))
        Vietnamese_dish.append(dish[0])
        price_list.append(prices[inde][0])

    return Vietnamese_dish, price_list

@app.route('/')
def home():
    return render_template('home.html')
# Health-checking method
@app.route('/healthCheck', methods=['GET'])
def health_check():
    """
    Health check the server
    Return:
    Status of the server
        "OK"
    """
    return "OK"



# Inference method
@app.route('/infer', methods=['POST'])
def infer():
    """
    Do inference on input image
    Return:
    Dictionary Object following this schema
        {
            "image_name": <Image Name>
            "infers":
            [
                {
                    "food_name_en": <Food Name in Englist>
                    "food_name_vi": <Food Name in Vietnamese>
                    "food_price": <Price of food>
                }
            ]
        }
    """

    # Read data from request
    time_api = time.time()
    image_name = request.form.get('image_name')
    encoded_img = request.form.get('image')

    # Convert base64 back to bytes
    # img = base64.b64decode(encoded_img)
    im_bytes = base64.b64decode(encoded_img)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)  

    # img = cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))

    # TODO
    # Call model for inference
    try:

        food_name_vi = []
        food_price = []
        time_segment = time.time()
        sub_segment_images =Segment(image=img)
        time_OCR = time.time()

        OCR_predictions = OCR(images=sub_segment_images)
        time_process = time.time()
        for sub_image in OCR_predictions:

            Vietnamese, price = readOCR_splitclass(sub_image[0],sub_image[1])
            Vietnamese_matched, price_matched = match_price(Vietnamese, price)
            food_name_vi.extend(Vietnamese_matched)
            food_price.extend(price_matched)


        food_price = convert_price(food_price)
        time_trans = time.time()
        food_name_en = transfer_English(food_name_vi)
        time_end = time.time()


        print("API transmit time: ",time_segment-time_api)
        print("Segment time: ",time_OCR - time_segment)
        print("OCR time: ", time_process-time_OCR)
        print("process time (match price + read OCR): ",time_trans-time_process)
        print("Transformer time: ",time_end-time_end)

        response = {
            "image_name": image_name,
            "infers": []
        }
        for pair in range(len(food_price)):
            dct = {
                'food_name_en': food_name_en[pair],
                'food_name_vi': food_name_vi[pair],
                'food_price': food_price[pair]
            }
            response['infers'].append(dct)

        return json.dumps(response)
        
    except:
        return None
    

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')

