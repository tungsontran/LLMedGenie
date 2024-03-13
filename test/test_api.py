import requests
import json

backend_url = "http://127.0.0.1:5000"

request_body = json.load(open("test/test_api_inference.json"))

response = requests.post(f"{backend_url}/inference/stream/", json=request_body)

for chunk in response.iter_lines():
    print(chunk.decode("utf-8"))
