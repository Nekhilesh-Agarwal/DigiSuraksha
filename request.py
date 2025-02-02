import requests
url = "https://crime-backend.onrender.com/api/auth/signup"
data = {
    "email": "nekhileshcr7@gmail.com",
    "password":"12345678",
}
response = requests.post(url, json=data)
print(response.json())  
