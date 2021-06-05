import base64
import os

upload_path = 'uploaded/images/'

def convert_to_image(images, token):
	images_path = []
	token_path = os.path.join(upload_path, token)

	if not os.path.exists(token_path):
		os.makedirs(token_path)

	for i in range(len(images)):
		if not isBase64(images[i]):
			continue

		file_name = "{}.jpg".format(i)
		temp_path = os.path.join(token_path, file_name)
		images_path.append(temp_path)

		with open(temp_path, "wb") as image_file:
			image_file.write(base64.b64decode(images[i]))

	return images_path

def isBase64(sb):
	try:
		if isinstance(sb, str):
			sb_bytes = bytes(sb, 'ascii')
		elif isinstance(sb, bytes):
			sb_bytes = sb
		else:
			return False

		return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes

	except Exception:
		return False