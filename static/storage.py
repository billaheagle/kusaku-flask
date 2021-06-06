from google.cloud import storage

BUCKET_NAME = 'bucketpredict'

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(file)

def send_to_storage(image_path):
   for path in image_path:
      destination_name = os.path.basename(path)
      upload_blob(BUCKET_NAME, path, destination_name)
