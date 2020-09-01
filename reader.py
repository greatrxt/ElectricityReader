try:
    from PIL import Image
except ImportError:
    import Image
import cv2
from PIL import ImageFilter
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# grey
# https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def process_img(img):
    # https://stackoverflow.com/questions/4632174/what-processing-steps-should-i-use-to-clean-photos-of-line-drawings/4632685#4632685
    white = img.filter(ImageFilter.BLUR).filter(ImageFilter.MaxFilter(15))
    grey = img.convert('L')
    width,height = img.size
    impix = img.load()
    whitepix = white.load()
    greypix = grey.load()
    for y in range(height):
        for x in range(width):
            greypix[x,y] = min(255, max(255 + impix[x,y][0] - whitepix[x,y][0], 255 + impix[x,y][1] - whitepix[x,y][1], 255 + impix[x,y][2] - whitepix[x,y][2]))


    img1 = change_contrast(grey, -10)

    thresh = 240
    fn = lambda x : 255 if x > thresh else 0
    img2 = img1.convert('L').point(fn, mode='1')
    return img2

def get_img():
    vcap = cv2.VideoCapture('rtsp://admin:Nalini1000@192.168.0.100:554/1')
    ret, frame = vcap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.rotate(-2)
    return img


def get_char_image(position, img):
    char_length = 63
    left = 363
    top = 370
    right = left + 375
    bottom = top + 90
    if position == 'all':
        img = img.crop((left, top, right, bottom))
    else:
        start = left + (char_length * (position))
        end = start + char_length
        img = img.crop((start, top, end, bottom))
        
    img2 = process_img(img)
    return img2


from tensorflow.keras.models import load_model
model = load_model('model.hdf5')


y_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'E', 'L', 'null']

def get_meter_reading():
    img = get_img()
    img_char_1 = np.array(get_char_image(0, img))
    img_char_2 = np.array(get_char_image(1, img))
    img_char_3 = np.array(get_char_image(2, img))
    img_char_4 = np.array(get_char_image(3, img))
    img_char_5 = np.array(get_char_image(4, img))
    img_char_6 = np.array(get_char_image(5, img))
    
    char_1 = y_classes[np.argmax(model.predict(np.expand_dims(img_char_1.flatten(), axis=0)))]
    if char_1 == 'null':
        char_1 = ''
    char_2 = y_classes[np.argmax(model.predict(np.expand_dims(img_char_2.flatten(), axis=0)))]
    if char_2 == 'null':
        char_2 = ''
    char_3 = y_classes[np.argmax(model.predict(np.expand_dims(img_char_3.flatten(), axis=0)))]
    if char_3 == 'null':
        char_3 = ''
    char_4 = y_classes[np.argmax(model.predict(np.expand_dims(img_char_4.flatten(), axis=0)))]
    char_5 = y_classes[np.argmax(model.predict(np.expand_dims(img_char_5.flatten(), axis=0)))]
    char_6 = y_classes[np.argmax(model.predict(np.expand_dims(img_char_6.flatten(), axis=0)))]
    if char_6 == 'null':
        char_6 = ''
    
    meter_reading = str(char_1) + str(char_2) + str(char_3) + str(char_4) + str(char_5) + str(char_6)
    
    return meter_reading


from datetime import datetime
from pytz import timezone    

india = timezone('Asia/Kolkata')
india_time = datetime.now(india)


from pathlib import Path

readings_file_name = 'readings.csv'
readings_file = Path(readings_file_name)
if readings_file.is_file():
    meter_readings = pd.read_csv(readings_file_name)
    print('Reading from disk')
else:
    meter_readings = pd.DataFrame(columns=['meter_reading', 'time', 'date'])
    meter_readings.to_csv(readings_file_name, index=False)
    print('Creating new file')
    
    
if meter_readings.size > 0:
    last_verified_reading = meter_readings.tail(1).iloc[0][0]
else:
    last_verified_reading = 21534


power_cons_file_name = 'kw.csv'
power_cons_file = Path(power_cons_file_name)
if power_cons_file.is_file():
    power_consumption = pd.read_csv(power_cons_file_name)
    print('Reading from disk')
else:
    power_consumption = pd.DataFrame(columns=['kw', 'time', 'date'])
    power_consumption.to_csv(power_cons_file_name, index=False)
    print('Creating new file')


print('Last meter reading is ' + str(last_verified_reading))


while True:
    meter_str = get_meter_reading()
    try:
        meter_reading = int(meter_str)

        #print(meter_reading)
        #print(len(meter_str))
        india_time = datetime.now(india)
        date = india_time.strftime('%H:%M:%S')
        time = india_time.strftime('%Y-%m-%d')

        #if last_verified_reading == meter_reading or meter_reading == last_verified_reading + 1:
        if meter_reading == last_verified_reading + 1:
            print("Meter reading : " + str(meter_reading) + " at " + india_time.strftime('%Y-%m-%d %H:%M:%S'))
            last_verified_reading = meter_reading 
            meter_readings.loc[meter_readings.size] = [meter_reading, date, time]
            meter_readings.to_csv(readings_file_name, mode='a', header=False, index=False)
            
            
        if len(meter_str) == 3: # for storing power consumtion in kw. Usually 3 characters
            #print("Power Consumption : " + str(meter_reading) + " at " + india_time.strftime('%Y-%m-%d %H:%M:%S'))
            power_consumption.loc[meter_readings.size] = [meter_reading, date, time]
            power_consumption.to_csv(power_cons_file_name, mode='a', header=False, index=False)
    except Exception as e:
        #print(e)
        pass
