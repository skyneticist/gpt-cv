import base64
import os
import cv2

from openai import OpenAI


def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode('utf-8')


def request_vision(image, update_message_callback):
    client = OpenAI()
    client.api_key = os.getenv("OPENAI_KEY")

    max_tokens = 300
    model = "gpt-4-vision-preview"
    base64_image = encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image? List each thing in a Markdown table in the first column and in the second column include the original country and era of origin for the idea behind each item found."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    # callback invoked
    update_message_callback()

    print(response.choices[0])
    return response.choices[0]
