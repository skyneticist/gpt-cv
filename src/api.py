from openai import OpenAI
from helpers import encode_image


def request_vision(image):
    client = OpenAI()
    client.api_key = ""
    # os.getenv("OPENAI_KEY")

    max_tokens = 300
    model = "gpt-4-vision-preview"
    base64_image = encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?\n List each thing in a Markdown table in the first column and in the second column include the original country and era of origin for the idea behind each item found."},
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
        model,
        messages,
        max_tokens,
    )

    print(response.choices[0])
    return response.choices[0]
