from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
from pymongo import MongoClient
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cdist
import base64
import os
from dotenv import load_dotenv

# Khởi tạo mô hình ResNet50
model2 = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img, xyxy):
    features = []
    for box in xyxy:
        # Chuyển đổi tọa độ từ [x1, y1, x2, y2] thành (left, upper, right, lower)
        x1, y1, x2, y2 = map(int, box)

        # Cắt ảnh
        cropped_img = img.crop((x1, y1, x2, y2))

        # Đảm bảo kích thước của ảnh cắt ra là (224, 224)
        cropped_img = cropped_img.resize((224, 224))

        # Chuyển đổi ảnh thành tensor
        img_array = image.img_to_array(cropped_img)
        img_array = np.expand_dims(img_array, axis=0)

        # Tiền xử lý ảnh cho mô hình ResNet50
        img_array = preprocess_input(img_array)

        # Đưa ảnh vào mô hình để lấy đặc trưng
        features_batch = model2.predict(img_array)
        features.append(features_batch.flatten())

    return features

app = Flask(__name__)
CORS(app)

model = YOLO("weight/lastv3.pt")

# ---MongoDB setup---
load_dotenv()
db_url = os.getenv("MONGO_URL")
mongo_client = MongoClient(db_url)
db = mongo_client['search_image']
collection = db['image_features']

names = ['Tshirt', 'dress', 'jacket', 'pants', 'shirt', 'short', 'skirt', 'sweater']

@app.route('/')
def home():
    return send_file('search_image.html')


@app.route('/diabetes/v1/predict', methods=['POST'])
def predict():
    # ---get the image from the request---
    image_file = request.files['image'].read()  # Lấy ảnh từ request
    img = Image.open(BytesIO(image_file))  # Chuyển đổi ảnh thành đối tượng PIL

    # ---process image using YOLO model---
    results = model(img)  # Sử dụng YOLO model để xử lý ảnh
    img_features = extract_features(img, results[0].boxes.xyxy)
    cls_array = results[0].boxes.cls.numpy()
    tags = [names[int(idx)] for idx in cls_array]
    conditions = [{'tags': tag} for tag in tags]
    query = {'$or': conditions}
    db_imgs = list(collection.find(query))
    db_features_list = []
    for db_img in db_imgs:
        features = db_img.get('features', [])
        db_img_tags = db_img.get('tags', [])
        # Lặp qua các feature trong 'features'
        for idx, feature in enumerate(features):
            if db_img_tags[idx] in tags:
                db_features_list.append(feature)
    img_features_array = np.array(img_features)
    db_features_array = np.array(db_features_list)
    distance_matrix = cdist(img_features_array, db_features_array, metric='euclidean')

    cnt = 0
    keyed_dict = {}
    for db_img in db_imgs:
        # Truy cập mảng 'features' thông qua key 'features'
        features = db_img.get('features', [])
        db_img_tags = db_img.get('tags', [])
        # Lặp qua các feature trong 'features'
        for idx, feature in enumerate(features):
            if db_img_tags[idx] in tags:
                keyed_dict[cnt] = db_img['filename']  # Truy cập thông qua key 'filename'
                cnt += 1

    sorted_indices = np.argsort(distance_matrix, axis=1)
    sorted_distances = np.take_along_axis(distance_matrix, sorted_indices, axis=1)
    list_product = []
    for column in sorted_indices:
        for element in column:
            if keyed_dict[element] not in list_product:
                list_product.append(keyed_dict[element])
    print(list_product)

    for db_img in db_imgs:
        print(db_img['filename'])
    for result in results:
        for box in result.boxes.xyxy:  # Lấy tọa độ xyxy của các bounding box
            x1, y1, x2, y2 = map(int, box)  # Chuyển tọa độ sang integer
            # Vẽ bounding box lên ảnh
            draw = ImageDraw.Draw(img)  # Khởi tạo đối tượng vẽ của PIL
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)  # Vẽ hình chữ nhật xanh

    # ---convert the processed image (with bounding boxes) back to byte array---
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')  # Lưu ảnh đã xử lý vào byte array
    img_byte_arr.seek(0)
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # ---send the processed image as a response---
    image_urls = [url_for('get_image', filename=img, _external=True) for img in list_product]
    print(image_urls)

    response_data = {
        'images': image_urls,
        'processed_image': f"data:image/png;base64,{img_base64}"  # Hình ảnh dưới dạng base64
    }
    return jsonify(response_data)

@app.route('/images/<filename>')
def get_image(filename):
    # Trả về ảnh từ server
    return send_file(f'image/product_image/{filename}', mimetype='image/png')


if __name__ == '__main__':
    app.run(port=5000)
