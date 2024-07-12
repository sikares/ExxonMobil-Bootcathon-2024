import requests
import json

# required only image url
# def recognize_url(image_url):
#     url = "https://carnet.ai/recognize-url"
#     payload = image_url
#     headers = {
#         'Accept': '*/*',
#         'Accept-Language': 'en-US,en;q=0.9,th;q=0.8',
#         'Connection': 'keep-alive',
#         'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
#         'DNT': '1',
#         'Origin': 'https://carnet.ai',
#         'Referer': 'https://carnet.ai/',
#         'Sec-Fetch-Dest': 'empty',
#         'Sec-Fetch-Mode': 'cors',
#         'Sec-Fetch-Site': 'same-origin',
#         'X-Requested-With': 'XMLHttpRequest',
#     }

#     response = requests.post(url, headers=headers, data=payload)
#     return response.json()

# image_url = "https%3A%2F%2Fcarnet.ai123132%2Fimg%2Ftiles%2F4.jpg"
# result = recognize_url(image_url)

# * Require both filename and binary file
def recognize_file(file_name, file_binary, file_type):
    url = "https://carnet.ai/recognize-file"
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,th;q=0.8",
        "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "Referer": "https://carnet.ai/",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

    files = {
        "imageFile": (file_name, file_binary, f"image/{file_type}")
    }
    

    response = requests.post(url, headers=headers, files=files)
    return response.json()

# with open("./q.jpg", "rb") as f:
#     file_binary = f.read()

# result = recognize_file("q.jpg", file_binary)

# if 'error' in result:
#     print("Error:", result['error'])
# else:
#     # Extract data if no error
#     car_info = result.get("car", {})
#     color_info = result.get("color", {})
#     angle_info = result.get("angle", {})
#     bbox_info = result.get("bbox", {})

#     print("Car Information:", car_info)
#     print("Color Information:", color_info)
#     print("Angle Information:", angle_info)
#     print("Bounding Box Information:", bbox_info)



# Example output of the request
# if error response will be 
    # {'error': 'Empty image provided'}


# if successresponse will be 
# {
#     "car": {
#         "make": "Peugeot",
#         "model": "207",
#         "generation": "I facelift (2009-2015)",
#         "years": "2009-2015",
#         "prob": "83.73"
#     },
#     "color": {
#         "name": "Silver",
#         "probability": 0.696
#     },
#     "angle": {
#         "name": "Front Right",
#         "probability": 0.5808
#     },
#     "bbox": {
#         "br_x": 0.9262,
#         "br_y": 0.8046,
#         "tl_x": 0.0619,
#         "tl_y": 0.1676
#     }
# }