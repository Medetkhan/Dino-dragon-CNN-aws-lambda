import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
from io import BytesIO
from urllib import request

interpreter = tflite.Interpreter(model_path='./dino-vs-dragon-v2.tflite') #./dino-vs-dragon.tflite
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = X / 255.0
    return X

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))
    X = preprocess(img)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds

#url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg"
#print(predict(url))

def lambda_handler(event, context):
    url = event['url']
    preds = predict(url)
    return {
        'statusCode': 200,
        'body':str(preds)
    }