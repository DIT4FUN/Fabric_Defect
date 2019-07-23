import requests
import time
import PIL.Image as Image
import base64
from io import BytesIO
from webapp.imgTool import readIMGInDir

for i in readIMGInDir("./webapp/test"):
    img = Image.open(i)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    times = time.time()
    res = requests.post('http://127.0.0.1:8000/api/', data={'image': base64_str, 'time': times})
    res.close()
    print(res.text)

