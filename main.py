from app import app
from flask import Flask, request, render_template, jsonify
from static.convert import convert_to_image
from static.predict import predict_image

IMAGES, TOKEN, RESULT = None, None, None

@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_request():
	global IMAGES, TOKEN, RESULT
	req_json = request.get_json()

	if 'images' not in req_json or 'token' not in req_json:
		return {'status': False, 'message': 'Missing parameter requirement (images, token)'}

	if type(req_json["token"]) is not str or type(req_json["images"]) is not list:
		return {'status': False,
				'message': 'Invalid paramter type => images must be List and token must be String'}

	TOKEN 		= req_json["token"]
	IMAGES 		= req_json["images"]
	images_path = convert_to_image(IMAGES, TOKEN)

	if len(images_path) == 0:
		return {'status': False, 'message': 'No image converted'}

	RESULT = predict_image(images_path)
	return {'status': True, 'images_path': images_path,
			'message': 'Successfully detect image', 'result': RESULT}

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
    app.run()