import socketio
import torch
from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import openai
from PIL import Image
import io

# Initialize Flask app and SocketIO
inference_server = Flask(__name__)
socketio = SocketIO(inference_server)

# Load the pre-trained tumor diagnosis model (PyTorch format)
tumor_diagnosis_model = torch.load('path_to_your_pytorch_model.pth')
tumor_diagnosis_model.eval()  # Set the model to evaluation mode

# Set OpenAI API key
openai.api_key = "your_openai_api_key"

def preprocess_image(image_data):
    """
    Preprocess the MRI image data for inference.
    Args:
        image_data: The MRI image data as bytes.
    Returns:
        A PyTorch tensor ready for inference.
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_data))
    # Resize and normalize the image
    image = image.resize((224, 224))  # Adjust size as per model requirements
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W) format
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image

def make_inference(image_data, diagnosis_id):
    """
    Perform inference on the MRI image data.
    Args:
        image_data: The MRI image data as bytes.
        diagnosis_id: The unique ID associated with the diagnosis.
    Returns:
        The predicted tumor type or 'tumor_free' if no tumor is detected.
    """
    try:
        # Preprocess the image
        image_tensor = preprocess_image(image_data)
        
        # Perform inference
        with torch.no_grad():
            output = tumor_diagnosis_model(image_tensor)
            _, predicted = torch.max(output, 1)
            tumor_name = predicted.item()  # Get the predicted class

        # Map the predicted class to a tumor type or 'tumor_free'
        if tumor_name == "no_tumor":  # Assuming class 0 corresponds to 'no tumor'
            return "tumor_free"
        else:
            return f"tumor_type_{tumor_name}"  # Replace with actual tumor types
    except Exception as e:
        print(f"Error during inference: {e}")
        return "tumor_free"  # Fallback to 'tumor_free' in case of errors

def get_tumor_info(tumor_name):
    """
    Fetch additional information about the diagnosed tumor using OpenAI's API.
    Args:
        tumor_name: The name of the diagnosed tumor.
    Returns:
        A string containing information about the tumor or an error message.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use the appropriate OpenAI model
            prompt=f"Tell me more about {tumor_name}.",
            max_tokens=150  # Limit the response length
        )
        return response.choices[0].text.strip()  # Extract the text response
    except Exception as e:
        return f"Error fetching tumor info: {str(e)}"  # Return error message

@socketio.on('make_inference')
def respond_to_inference_request(data):
    """
    Handle incoming WebSocket requests for tumor diagnosis.
    Args:
        data: A dictionary containing 'image_data' and 'diagnosis_id'.
    """
    try:
        image_data = data['image_data']
        diagnosis_id = data['diagnosis_id']
        tumor_name = make_inference(image_data, diagnosis_id)

        if tumor_name == "tumor_free":
            emit('inference_result', {
                'diagnosis': tumor_name,
                'info': 'No tumor detected',
                'ID': diagnosis_id
            })
        else:
            tumor_info = get_tumor_info(tumor_name)
            emit('inference_result', {
                'diagnosis': tumor_name,
                'info': tumor_info,
                'ID': diagnosis_id
            })
    except Exception as e:
        emit('error', {'error': str(e)})

if __name__ == '__main__':
    # Start the Flask server with WebSocket support
    socketio.run(inference_server, host='0.0.0.0', port=5000)