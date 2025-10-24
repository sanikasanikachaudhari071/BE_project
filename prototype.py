"""
Deepfake Detection System Prototype
Based on the system architecture diagram with dual-stream processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import cv2
from mtcnn import MTCNN
from scipy.fft import dct

class PreprocessingModule:
    """Preprocessing module using MTCNN for face detection and alignment"""
    
    def __init__(self):
        self.mtcnn = MTCNN()
    
    def preprocess(self, image):
        """
        Preprocess input image/video frame
        - Crop and align face using MTCNN
        - Resize to 224x224
        """
        # Detect face using MTCNN
        result = self.mtcnn.detect_faces(image)
        
        if len(result) == 0:
            # If no face detected, return resized original image
            processed = cv2.resize(image, (224, 224))
            return torch.tensor(processed).permute(2, 0, 1).float() / 255.0
        
        # Get the first detected face
        face_box = result[0]['box']
        x, y, w, h = face_box
        
        # Crop face
        face_crop = image[y:y+h, x:x+w]
        
        # Resize to 224x224
        face_resized = cv2.resize(face_crop, (224, 224))
        
        # Convert to tensor and normalize
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255.0
        
        return face_tensor

class SpatialStream(nn.Module):
    """Spatial stream using DenseNet121 for feature extraction"""
    
    def __init__(self):
        super(SpatialStream, self).__init__()
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(pretrained=True)
        # Remove the classifier to get features
        self.densenet.classifier = nn.Identity()
        # Add a projection layer to get spatial vector
        self.projection = nn.Linear(1024, 512)
        
        # Initialize projection layer with better weights
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, x):
        # Extract features using DenseNet121
        features = self.densenet(x)
        # Project to spatial vector
        spatial_vector = self.projection(features)
        return spatial_vector

class FrequencyStream(nn.Module):
    """Frequency stream using DCT and CNN for frequency domain analysis"""
    
    def __init__(self):
        super(FrequencyStream, self).__init__()
        # CNN for processing frequency domain data
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # 224 -> 112 -> 56 -> 28
        self.fc = nn.Linear(256 * 28 * 28, 512)
    
    def dct_transform(self, x):
        """Apply Discrete Cosine Transform to convert to frequency domain"""
        # Convert to numpy for DCT
        x_np = x.detach().cpu().numpy()
        dct_result = np.zeros_like(x_np)
        
        for i in range(x_np.shape[0]):  # batch dimension
            for j in range(x_np.shape[1]):  # channel dimension
                dct_result[i, j] = dct(dct(x_np[i, j], axis=0, norm='ortho'), axis=1, norm='ortho')
        
        return torch.tensor(dct_result, device=x.device, dtype=x.dtype)
    
    def forward(self, x):
        # Apply DCT transform
        freq_x = self.dct_transform(x)
        
        # Process through CNN
        x = F.relu(self.conv1(freq_x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten and project to frequency vector
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        freq_vector = self.fc(x)
        
        return freq_vector

class FeatureFusion(nn.Module):
    """Feature fusion module to combine spatial and frequency vectors"""
    
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.fusion_layer = nn.Linear(512 + 512, 512)  # Concatenate and project
    
    def forward(self, spatial_vector, freq_vector):
        # Concatenate spatial and frequency vectors
        fused_features = torch.cat([spatial_vector, freq_vector], dim=1)
        # Project to unified feature space
        fused_output = self.fusion_layer(fused_features)
        return fused_output

class CrossViT(nn.Module):
    """Simplified Cross Vision Transformer for analysis"""
    
    def __init__(self, input_dim=512, hidden_dim=256, num_heads=8, num_layers=2):
        super(CrossViT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feed forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Apply transformer layers
        for attention, layer_norm, ffn in zip(self.attention_layers, self.layer_norms, self.ffns):
            # Self-attention
            attn_output, _ = attention(x, x, x)
            x = layer_norm(x + attn_output)
            
            # Feed forward
            ffn_output = ffn(x)
            x = layer_norm(x + ffn_output)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        return x

class ClassificationHead(nn.Module):
    """MLP head with sigmoid activation for binary classification"""
    
    def __init__(self, input_dim=256):
        super(ClassificationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights properly
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.mlp(x)

class DeepfakeDetectionSystem(nn.Module):
    """Complete deepfake detection system implementing the architecture"""
    
    def __init__(self):
        super(DeepfakeDetectionSystem, self).__init__()
        
        # Initialize all components
        self.preprocessing = PreprocessingModule()
        self.spatial_stream = SpatialStream()
        self.frequency_stream = FrequencyStream()
        self.feature_fusion = FeatureFusion()
        self.transformer = CrossViT()
        self.classification_head = ClassificationHead()
    
    def forward(self, x):
        # Preprocessing (assuming x is already preprocessed tensor)
        # In real implementation, you would call self.preprocessing.preprocess(x) here
        
        # Dual stream processing
        spatial_vector = self.spatial_stream(x)
        freq_vector = self.frequency_stream(x)
        
        # Feature fusion
        fused_features = self.feature_fusion(spatial_vector, freq_vector)
        
        # Transformer analysis
        transformer_output = self.transformer(fused_features)
        
        # Classification
        prediction = self.classification_head(transformer_output)
        
        return prediction
    
    def preprocess_input(self, image):
        """Preprocess input image using MTCNN"""
        return self.preprocessing.preprocess(image)

def create_prototype():
    """Create and return the prototype model"""
    model = DeepfakeDetectionSystem()
    return model

def load_image(image_path):
    """Load and preprocess an image from file path"""
    try:
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Loaded image: {image_path}")
        print(f"Original image shape: {image.shape}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_image(model, image_path):
    """Predict if an image is real or fake"""
    print(f"\nAnalyzing image: {image_path}")
    print("-" * 50)
    print("⚠️  WARNING: This is an UNTRAINED prototype model!")
    print("   Results are NOT reliable - model needs training on real/fake data.")
    print("-" * 50)
    
    # Load the image
    image = load_image(image_path)
    if image is None:
        return
    
    # Preprocess the image
    print("Preprocessing image...")
    preprocessed = model.preprocess_input(image)
    print(f"Preprocessed shape: {preprocessed.shape}")
    
    # Add batch dimension and predict
    input_tensor = preprocessed.unsqueeze(0)  # Add batch dimension
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Forward pass
    print("Running deepfake detection...")
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Interpret results
    fake_probability = prediction.item()
    real_probability = 1 - fake_probability
    
    print(f"\nResults:")
    print(f"Fake probability: {fake_probability:.4f} ({fake_probability*100:.2f}%)")
    print(f"Real probability: {real_probability:.4f} ({real_probability*100:.2f}%)")
    
    if fake_probability > 0.7:
        result = "FAKE (High confidence)"
    elif fake_probability > 0.5:
        result = "Likely FAKE"
    elif fake_probability > 0.3:
        result = "Likely REAL"
    else:
        result = "REAL (High confidence)"
    
    print(f"Prediction: {result}")
    return fake_probability

def test_prototype():
    """Test the prototype with user input"""
    print("Creating Deepfake Detection System Prototype...")
    
    # Create model
    model = create_prototype()
    print("Model loaded successfully!")
    
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION SYSTEM")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Test with an image file")
        print("2. Test with dummy data")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter the path to your image file: ").strip()
            if image_path:
                predict_image(model, image_path)
            else:
                print("Please enter a valid image path.")
        
        elif choice == "2":
            print("\nTesting with dummy data...")
            # Create dummy input (batch_size=1, channels=3, height=224, width=224)
            dummy_input = torch.randn(1, 3, 224, 224)
            
            print(f"Input shape: {dummy_input.shape}")
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"Output shape: {output.shape}")
            print(f"Prediction (fake probability): {output.item():.4f}")
            
            # Test preprocessing module separately
            print("\nTesting preprocessing module...")
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            preprocessed = model.preprocess_input(dummy_image)
            print(f"Preprocessed shape: {preprocessed.shape}")
        
        elif choice == "3":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\nPrototype created successfully!")
    print("Architecture components:")
    print("- Preprocessing: MTCNN for face detection and alignment")
    print("- Spatial Stream: DenseNet121 for spatial feature extraction")
    print("- Frequency Stream: DCT + CNN for frequency domain analysis")
    print("- Feature Fusion: Concatenation and projection")
    print("- Transformer: Cross-ViT for analysis")
    print("- Classification: MLP head with sigmoid activation")

if __name__ == "__main__":
    test_prototype()
