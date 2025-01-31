# WhatsApp Chatbot using FastAPI and Twilio

This is a **WhatsApp chatbot** built with **FastAPI** and integrated with **GPT-4o** via OpenAI, using **Twilio's API** to send and receive messages.

## 🚀 Features
- **Receives messages from WhatsApp via Twilio**
- **Processes messages using ChatGPT (GPT-4o)**
- **Sends AI-generated responses back to WhatsApp**
- **Uses Twilio's TwiML for message handling**

## 🛠️ Setup & Installation

### **1️⃣ Install Dependencies**
```sh
pip install fastapi uvicorn pydantic-ai twilio python-multipart
```

### **2️⃣ Set Up Twilio Sandbox for WhatsApp**
1. **Go to Twilio Console** → [Twilio WhatsApp Sandbox](https://www.twilio.com/console/sms/whatsapp/sandbox)
2. **Activate the sandbox** and note the WhatsApp number.
3. **Set the webhook URL**:
   ```
   https://your-ngrok-url.ngrok-free.app/chat
   ```
4. **Ensure the method is `POST`**.

### **3️⃣ Run the FastAPI Server**
```sh
uvicorn whatsapp_bot:app --host 0.0.0.0 --port 8000 --reload
```

### **4️⃣ Expose Localhost to the Internet (Ngrok)**
Twilio needs a public URL to reach your local server. Run:
```sh
ngrok http 8000
```
This will generate a URL like:
```
https://7dd5-201-218-255-109.ngrok-free.app
```
Use this **Ngrok URL** as the webhook in **Twilio Console**.

### **5️⃣ Test the Bot**
- Send a message to the Twilio **sandbox number**.
- Check FastAPI logs for received messages.
- You should receive an **AI-generated response** in WhatsApp!

## 🐞 Troubleshooting
### **1️⃣ Check Twilio Debugger**
If responses aren’t sent, visit:
[Twilio Debugger](https://www.twilio.com/console/debugger) to check for errors.

### **2️⃣ Verify Webhook Setup**
Ensure Twilio's **Webhook URL** is correctly set to `https://your-ngrok-url.ngrok-free.app/chat` with **POST**.

### **3️⃣ Restart Services**
If messages aren’t processed:
```sh
uvicorn whatsapp_bot:app --host 0.0.0.0 --port 8000 --reload
```
```sh
ngrok http 8000
```
Then update the **Twilio webhook** with the new Ngrok URL.

---