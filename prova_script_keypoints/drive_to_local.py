from Google import Create_Service
import os
import io 
from googleapiclient.http import MediaIoBaseDownload
import google.auth
import googleapiclient.discovery
import googleapiclient.errors
import shutil

CLIENT_SECRET_FILE = 'client_secret_GoogleCloud.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

#print(dir(service))

parent_folder_id = '1Hu7Oxk82OvpCXmzK-3A2BttWw_us88ow'
local_directory = '/home/coloranto/Desktop/tesi/prova_script_keypoints/all_video_keypoints/'


# Recursively download the contents of the parent folder
def download_folder(folder_id, local_path):
    # Query for the list of files and folders in the current folder
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name, mimeType)").execute()
    items = results.get("files", [])
    # Download each file and recursively call this function for each folder
    for item in items:
        file_id = item['id']
        file = service.files().get(fileId=file_id).execute()
        file_name = file['name']
        if 'folder' in file['mimeType']:
            # Recursively call this function for the folder
            new_local_path = f'{local_path}/{file_name}'
            os.makedirs(new_local_path, exist_ok=True)
            download_folder(file_id, new_local_path)
        else:
            # Download the file
            print(f'Downloading file: {local_path}/{file_name}')
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = googleapiclient.http.MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)
            # Save the file to the local directory
            with open(f'{local_path}/{file_name}', 'wb') as f:
                f.write(fh.read())

# Create the local directory and start the recursive download
os.makedirs(local_directory, exist_ok=True)
download_folder(parent_folder_id, local_directory)



