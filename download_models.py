import os
import requests

from config import CHECKPOINT_DIR

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

print('Dowloading Trained Model (63Mb)...')
download_file_from_google_drive('1mvgGwj-ShSv1Y0pjPDXsjDqZLRYeY7g5', CHECKPOINT_DIR + '/checkpoint')
download_file_from_google_drive('1e8wETHJy2NqFz3uXUaKZkoMTQ6fZdkTX', CHECKPOINT_DIR + '/model.ckpt.index')
download_file_from_google_drive('1J7gw9-1XCFceITeNQsg5FMysfM-YSk5I', CHECKPOINT_DIR + '/model.ckpt.meta')
download_file_from_google_drive('1WwzMPHulDxAyFYqCmyXeVWrWfPqE2Xc-', CHECKPOINT_DIR + '/model.ckpt.data-00000-of-00001')
print('Done.')