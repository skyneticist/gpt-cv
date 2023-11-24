import base64
import cv2
import json

from api import request_vision


def make_vision_request(img, callback):
    handle = request_vision(img)
    callback(handle)


def encode_image(image_path):
    _, buffer = cv2.imencode(".jpg", image_path)
    return base64.b64encode(buffer).decode('utf-8')


def save(response):
    # Define the path to the desired content
    path_to_content = ["Choice", "message",
                       "ChatCompletionMessage", "content"]

    # Extract the content
    extracted_content = extract_content_from_json(
        response, path_to_content)

    # Save the content as a Markdown file, if extraction was successful
    if extracted_content is not None:
        save_markdown("extracted_content.md", extracted_content)
        print("Markdown content saved successfully.")
    else:
        print("Failed to extract content.")


def extract_content_from_json(json_str, path):
    """
    Extract content from a JSON string given a path to the desired key.

    :param json_str: JSON string
    :param path: List of keys leading to the desired content
    :return: Extracted content or None if any key is not found
    """
    try:
        data = json.loads(json_str)
        for key in path:
            data = data[key]
        return data
    except KeyError as e:
        print(f"Key not found: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return None


def save_markdown(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
