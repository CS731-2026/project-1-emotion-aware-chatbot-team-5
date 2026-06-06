import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import cv2

# Label order must match how the model was trained.
# This order matches the working `yolov8_face.py` script.
LABELS = ['anger', 'happy', 'surprise', 'sad', 'contempt', 'fear', 'disgust', 'neutral']
NORMALIZE_LABEL = {
    'anger': 'angry',
}
EMOJI_MAP = {
    'happy': '😊', 'sad': '😢', 'angry': '😠',
    'fear': '😨', 'surprise': '😲', 'disgust': '🤢',
    'contempt': '😒', 'neutral': '😐'
}

class Inferencer:
    def __init__(self, model_path):
        # Determine the device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pre-trained model
        self.model = self.load_model(model_path)
        
        # Define the image transformation pipeline. These MUST be the same as the training transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            # Normalize for 3 channels (RGB)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_model(self, model_path):
        # Load the entire model from the specified path
        # map_location ensures the model is loaded to the correct device (CPU or GPU)
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Set the model to evaluation mode (important for inference)
        model.eval()
        return model
    
    def preprocess_image(self, image):
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        image = Image.fromarray(image)
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)
    


    def predict(self, image_path):
        """
        Predict emotion for an input image (numpy BGR image expected).
        Returns: (label, display, confidence, emoji)
        """
        image = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Map index -> label/display/emoji
        try:
            label = LABELS[predicted_class]
        except Exception:
            label = 'neutral'
        label = NORMALIZE_LABEL.get(label, label)
        display = label.capitalize()
        emoji = EMOJI_MAP.get(label, '😐')
        return label, display, confidence, emoji

# Usage example:
if __name__ == "__main__":
    # Specify the path to your saved model
    model_path = "checkpoints/8.pt"  # Adjust this to your saved model path
    
    # Create an instance of the Inferencer class
    inferencer = Inferencer(model_path)

    # Specify the path to the image you want to classify
    image_path = "test_images/surprise/ffhq_24.png"
    
    # Perform the prediction
    image = cv2.imread(image_path)
    predicted_class, confidence = inferencer.predict(image)
    
    # Print the results
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")