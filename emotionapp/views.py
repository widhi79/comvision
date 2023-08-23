from django.shortcuts import render
from .deteksiemosi import ModulDeteksiEmosi

# Create your views here.
import cv2
import numpy as np
from django.conf import settings
import os
import asyncio


def detect_emotion_from_image(image):
    module = ModulDeteksiEmosi()
    results = []

    img_array = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    results = module.process_frame(img)

    return results[0][4] if results else None


def save_uploaded_file(uploaded_file):
    # Construct the destination path where the file will be saved
    file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

    if not os.path.exists(file_path):
        # Save the uploaded file to the specified path
        with open(file_path, "wb") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

    return file_path


def delete_file_async(file_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def delete_file():
        await asyncio.sleep(10)  # Delay to ensure the image is displayed

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File deleted: {file_path}")

    loop.run_until_complete(delete_file())
    loop.close()


def index(request):
    emotion = None
    uploaded_image = None

    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        emotion = detect_emotion_from_image(image)

        # Read the contents of the InMemoryUploadedFile and encode as Base64
        image_contents = image.read()
        # uploaded_image = base64.b64encode(image_contents).decode("utf-8")

        uploaded_image = os.path.join(settings.MEDIA_URL, image.name)
        print("Uploaded Image:", uploaded_image)  # Print the encoded image data
        saved_file_path = save_uploaded_file(image)

        # delete_file_async(saved_file_path)

    return render(
        request, "index.html", {"emotion": emotion, "uploaded_image": uploaded_image}
    )
