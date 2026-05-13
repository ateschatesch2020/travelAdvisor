from dotenv import load_dotenv
import requests
import os
load_dotenv()

header = {"Authorization": f"Bearer {os.environ["OPENROUTER_API_KEY"]}"}
url = "https://openrouter.ai/api/v1/models/user"
response = requests.get(url, headers=header)

data = response.json()["data"]
print(data)