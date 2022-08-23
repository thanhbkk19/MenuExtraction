import json
from flask import Flask, request, json
import base64
from Segment.test_segment import Segment
import cv2
from CRAFT_OCR.src import OCR, readOCR_splitclass
from Transformer.main_transformer import transfer_English
import glob
import re

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


food_name_vi = []
food_price = []

img = "001.png"
image_name = "001.png"
image = cv2.imread(img)
Segment(img,img_name=image_name, image=image, mode = "image")
OCR_path = "val_test"
OCR(image_url=OCR_path)
for sub_image in glob.glob("val_test"+"/*.png"):
    Vietnamese, price = readOCR_splitclass(sub_image)
    Vietnamese_matched, price_matched = match_price(Vietnamese, price)
    # print(Vietnamese_matched)
    # print(price_matched)
    food_name_vi.extend(Vietnamese_matched)
    food_price.extend(price_matched)

food_price = convert_price(food_price)
food_name_en = transfer_English(food_name_vi)

