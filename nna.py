import torch
import torch.nn as nn

class Brain2TextModel(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim=128):
        """
        Neural network architecture for decoding brain signals into text.

        Args:
            input_channels (int): Number of input channels (e.g., EEG/MEG sensors).
            num_classes (int): Number of output classes (e.g., characters or tokens).
            hidden_dim (int): Hidden dimension size for the RNN/Transformer.
        """
        super(Brain2TextModel, self).__init__()
        
        # Convolutional Neural Network (CNN) for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Recurrent Neural Network (LSTM) for temporal sequence modeling
        self.rnn = nn.LSTM(
            input_size=64,  # Input size from CNN output
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # Bidirectional LSTM doubles hidden_dim
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Output size corresponds to number of classes
        )
    
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, sequence_length, num_classes).
        """
        # CNN: Extract spatial features
        batch_size = x.size(0)
        x = self.cnn(x)  # Shape: (batch_size, channels_out, height_out, width_out)
        
        # Reshape for RNN: Flatten spatial dimensions into temporal sequences
        x = x.permute(0, 3, 1, 2)  # Shape: (batch_size, width_out, channels_out, height_out)
        x = x.flatten(2)           # Shape: (batch_size, width_out, channels_out * height_out)
        
        # RNN: Temporal modeling
        x_rnn_out, _ = self.rnn(x)  # Shape: (batch_size, sequence_length, hidden_dim * 2)
        
        # Fully connected layers
        logits = self.fc(x_rnn_out)  # Shape: (batch_size, sequence_length, num_classes)
        
        return logits

# Example Usage
if __name__ == "__main__":
    # Define model parameters
    input_channels = 16   # Number of EEG/MEG sensors
    num_classes = 27      # Number of output classes (26 letters + space)
    batch_size = 8
    sequence_length = 100
    height = 32           # Height of input signal representation
    width = sequence_length

    # Create the model
    model = Brain2TextModel(input_channels=input_channels, num_classes=num_classes)

    # Example input tensor: Simulated EEG/MEG data
    example_input = torch.rand(batch_size, input_channels, height, width)

    # Forward pass through the model
    output_logits = model(example_input)

    print(f"Output shape: {output_logits.shape}")  # Expected shape: (batch_size, sequence_length, num_classes)
