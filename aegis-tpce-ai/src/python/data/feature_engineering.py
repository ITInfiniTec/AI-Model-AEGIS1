# src/python/data/feature_engineering.py

import logging
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineeringConfig(BaseModel):
    """
    Configuration for the FeatureEngineer.
    This allows for type-safe and validated setup of the feature engineering pipeline.
    """
    enabled_features: List[str] = Field(
        default=["kinematic", "spectral", "signature"],
        description="List of feature types to extract. Options: 'kinematic', 'spectral', 'signature'."
    )
    sampling_rate: float = Field(
        default=1000.0,
        gt=0,
        description="Sampling rate of the sensor data in Hz. Required for kinematic and spectral features."
    )
    use_gpu: bool = Field(
        default=torch.cuda.is_available(),
        description="Flag to use GPU for tensor operations."
    )

class FeatureEngineer:
    """
    A modular, pipeline-compatible class for transforming raw time-series sensor data
    into meaningful feature vectors suitable for machine learning models.
    """
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.device = torch.device("cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu")
        logging.info(f"FeatureEngineer initialized. Using device: {self.device}")
        logging.info(f"Enabled features: {self.config.enabled_features}")

    def _extract_kinematic_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Extracts kinematic features (velocity, acceleration) from time-series data.
        This method is GPU-accelerated.
        """
        logging.debug(f"Extracting kinematic features from tensor of shape {data.shape}")

        # Ensure data is on the correct device
        data = data.to(self.device)

        # Calculate derivatives using torch.diff
        # Note: torch.diff reduces the sequence length by n.
        # We will pad the results with zeros to maintain the original sequence length.

        # Velocity (1st derivative)
        velocity = torch.diff(data, n=1, dim=2) * self.config.sampling_rate
        # Pad with one zero at the beginning of the sequence dimension
        velocity = F.pad(velocity, (1, 0), "constant", 0)

        # Acceleration (2nd derivative)
        acceleration = torch.diff(data, n=2, dim=2) * (self.config.sampling_rate ** 2)
        # Pad with two zeros at the beginning of the sequence dimension
        acceleration = F.pad(acceleration, (2, 0), "constant", 0)

        # Concatenate original data, velocity, and acceleration along the channel dimension
        # This creates a richer feature set for each time step.
        # The output shape will be (batch_size, num_channels * 3, sequence_length)
        kinematic_features = torch.cat((data, velocity, acceleration), dim=1)

        logging.debug(f"Kinematic features extracted. Output shape: {kinematic_features.shape}")
        return kinematic_features

    def _extract_spectral_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Extracts spectral features from the signal using FFT.
        This method is GPU-accelerated.
        """
        logging.debug(f"Extracting spectral features from tensor of shape {data.shape}")

        # Ensure data is on the correct device
        data = data.to(self.device)
        batch_size, num_channels, seq_len = data.shape

        # Compute the one-dimensional, real-valued Fast Fourier Transform
        fft_result = torch.fft.rfft(data, n=seq_len, dim=2)
        magnitude_spectrum = torch.abs(fft_result)

        # Get the corresponding frequencies
        frequencies = torch.fft.rfftfreq(n=seq_len, d=1.0 / self.config.sampling_rate, device=self.device)

        # --- Feature Derivation ---

        # 1. Dominant Frequency
        dominant_freq_indices = torch.argmax(magnitude_spectrum, dim=2)
        dominant_freq = frequencies[dominant_freq_indices]

        # 2. Spectral Centroid (Center of Mass of the Spectrum)
        # Add epsilon for numerical stability
        spectral_centroid = torch.sum(frequencies * magnitude_spectrum, dim=2) / (torch.sum(magnitude_spectrum, dim=2) + 1e-9)

        # 3. Spectral Entropy (Measure of signal complexity)
        power_spectral_density = magnitude_spectrum.pow(2)
        psd_normalized = power_spectral_density / (torch.sum(power_spectral_density, dim=2, keepdim=True) + 1e-9)
        # Add epsilon inside the log for numerical stability
        spectral_entropy = -torch.sum(psd_normalized * torch.log2(psd_normalized + 1e-9), dim=2)

        # --- Feature Concatenation ---
        # Stack the derived scalar features for each channel
        # The shape of each feature is (batch_size, num_channels)
        # We stack them to get (batch_size, num_channels, 3)
        spectral_features = torch.stack([dominant_freq, spectral_centroid, spectral_entropy], dim=2)

        # Flatten the channel and feature dimensions to (batch_size, num_channels * 3)
        spectral_features_flat = spectral_features.view(batch_size, -1)

        logging.debug(f"Spectral features extracted. Output shape: {spectral_features_flat.shape}")
        return spectral_features_flat

    def _extract_signature_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Extracts statistical signature features from the time-domain signal.
        This method is GPU-accelerated.
        """
        logging.debug(f"Extracting signature features from tensor of shape {data.shape}")

        # Ensure data is on the correct device
        data = data.to(self.device)
        batch_size, num_channels, seq_len = data.shape

        # --- Feature Calculation (along the time axis, dim=2) ---

        # 1. Mean
        mean = torch.mean(data, dim=2)

        # 2. Standard Deviation
        std = torch.std(data, dim=2)

        # 3. Skewness (Third Statistical Moment)
        # For stable computation, we calculate the z-score first
        mean_expanded = mean.unsqueeze(2) # for broadcasting
        std_expanded = std.unsqueeze(2)   # for broadcasting
        z = (data - mean_expanded) / (std_expanded + 1e-6) # Epsilon for stability
        skew = torch.mean(z**3, dim=2)

        # 4. Kurtosis (Fourth Statistical Moment)
        kurt = torch.mean(z**4, dim=2)

        # 5. Root Mean Square (RMS)
        rms = torch.sqrt(torch.mean(data**2, dim=2))

        # --- Feature Concatenation ---
        # Stack the derived scalar features for each channel
        # The shape of each feature is (batch_size, num_channels)
        # We stack them to get (batch_size, num_channels, 5)
        signature_features = torch.stack([mean, std, skew, kurt, rms], dim=2)

        # Flatten the channel and feature dimensions to (batch_size, num_channels * 5)
        signature_features_flat = signature_features.view(batch_size, -1)

        logging.debug(f"Signature features extracted. Output shape: {signature_features_flat.shape}")
        return signature_features_flat

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies the full feature engineering pipeline to the input data.

        Args:
            data (torch.Tensor): A batch of time-series data with shape
                                 (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: A tensor of engineered features, ready for a model.
        """
        data = data.to(self.device)
        logging.info(f"Starting feature engineering transform on tensor of shape {data.shape}")

        all_features: List[torch.Tensor] = []

        if "kinematic" in self.config.enabled_features:
            kinematic_features = self._extract_kinematic_features(data)
            # Flatten the sequence and channel dimensions for concatenation
            all_features.append(kinematic_features.view(kinematic_features.shape[0], -1))

        if "spectral" in self.config.enabled_features:
            spectral_features = self._extract_spectral_features(data)
            all_features.append(spectral_features)

        if "signature" in self.config.enabled_features:
            signature_features = self._extract_signature_features(data)
            all_features.append(signature_features)

        if not all_features:
            logging.warning("No features were extracted. Returning an empty tensor.")
            return torch.empty((data.shape[0], 0), device=self.device)

        # Concatenate all extracted features along the feature dimension
        combined_features = torch.cat(all_features, dim=1)
        logging.info(f"Feature engineering complete. Output shape: {combined_features.shape}")

        return combined_features

if __name__ == '__main__':
    # Example usage

    # 1. Configuration
    fe_config = FeatureEngineeringConfig(
        enabled_features=["kinematic", "spectral", "signature"],
        sampling_rate=1024.0
    )

    # 2. Initialization
    feature_engineer = FeatureEngineer(fe_config)

    # 3. Create dummy input data
    batch_size = 4
    num_channels = 1 # e.g., a single radar signal
    sequence_length = 2048
    dummy_data = torch.randn(batch_size, num_channels, sequence_length)

    # 4. Run the transformation
    # In a real pipeline, this dummy_data would come from the SensorDataLoader
    engineered_features = feature_engineer.transform(dummy_data)

    print(f"\nSuccessfully transformed data.")
    print(f"Initial data shape: {dummy_data.shape}")
    print(f"Engineered features shape: {engineered_features.shape}")
