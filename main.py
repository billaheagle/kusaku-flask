import os
import shutil
from flask import Flask, request, render_template, jsonify
from static.convert import convert_to_image
from static.predict import predict_image
from static.storage import send_to_storage

IMAGES, TOKEN, RESULT = None, None, None
app=Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_request():
    global IMAGES, TOKEN, RESULT
    try:
        req_json = request.get_json()
    except:
        return {'status': False, 'message': 'Cant read JSON'}

    try:
        if 'images' not in req_json or 'token' not in req_json:
            return {'status': False, 'message': 'Missing parameter requirement (images, token)'}

        if type(req_json["token"]) is not str or type(req_json["images"]) is not list:
            return {'status': False, 'message': 'Invalid paramter type => images must be List and token must be String'}
    except:
        return {'status': False, 'message': 'Cant Identify JSON'}

    try:
        TOKEN       = req_json["token"]
        IMAGES      = req_json["images"]
        images_path = convert_to_image(IMAGES, TOKEN)

        if len(images_path) == 0:
            return {'status': False, 'message': 'No image converted'}
    except:
        return {'status': False, 'message': 'Cant process JSON'}

    try:
        RESULT = predict_image(images_path)
    except:
        return {'status': False, 'message': 'Cant predict JSON'}

    try:
        send_to_storage(image_path)
    except:
        print('cant send to storage')

    upload_path = 'uploaded/images/'
    token_path = os.path.join(upload_path, TOKEN)

    if os.path.exists(token_path):
        shutil.rmtree(token_path)

    return {'status': True, 'images_path': images_path, 'message': 'Successfully detect image', 'result': RESULT}

@app.route('/result', methods=['GET'])
def send_result():
    global IMAGES, TOKEN, RESULT
    token = request.args.get('token')

    if token is None:
        return {'status': False, 'message': 'Missing parameter requirement token'}

    if TOKEN is None or IMAGES is None:
        return {'status': False, 'message': 'You have not uploaded an image and predict it'}

    if token != TOKEN:
        return {'status': False, 'message': 'Token does not match'}

    if RESULT is None:
        return {'status': False, 'message': 'Image prediction is still in progress'}

    temp_result = RESULT
    IMAGES, TOKEN, RESULT = None, None, None

    return {'status': True, 'message': 'Successfully detect and predict image and your prediction result has been deleted from the system', 'result': temp_result}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
