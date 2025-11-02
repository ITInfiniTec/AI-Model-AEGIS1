# src/python/data/sensor_data_loader.py

import logging
from enum import Enum
from typing import Dict, Any, List, Generator, Tuple

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SensorType(str, Enum):
    """Enumeration for different types of AEGIS sensors."""
    RADAR = "RADAR"
    SONAR = "SONAR"
    ELECTRO_OPTICAL = "ELECTRO_OPTICAL"
    INFRARED = "INFRARED"
    ELECTRONIC_WARFARE = "ELECTRONIC_WARFARE"

class SensorDataValidator(BaseModel):
    """
    Pydantic model for validating the structure and content of a single sensor data record.
    Ensures data integrity before it enters the processing pipeline.
    """
    timestamp: float = Field(..., description="UNIX timestamp of the sensor reading.")
    sensor_id: str = Field(..., description="Unique identifier for the sensor.")
    sensor_type: SensorType = Field(..., description="Type of the sensor.")
    data: List[float] = Field(..., description="Raw sensor measurement data.")
    metadata: Dict[str, Any] = Field({}, description="Optional metadata.")

    @validator('data')
    def data_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Sensor data list cannot be empty')
        return v

class SensorDataLoaderConfig(BaseModel):
    """Configuration for the SensorDataLoader."""
    file_path: str = Field(..., description="Path to the sensor data file (CSV, JSON, etc.).")
    batch_size: int = Field(default=64, gt=0, description="Number of records to load per batch.")
    use_gpu: bool = Field(default=torch.cuda.is_available(), description="Flag to use GPU for tensors.")

class SensorDataLoader:
    """
    A robust, efficient, and modular component for loading and preprocessing
    simulated AEGIS sensor data.
    """
    def __init__(self, config: SensorDataLoaderConfig):
        self.config = config
        self.device = torch.device("cuda" if self.config.use_gpu else "cpu")
        logging.info(f"SensorDataLoader initialized. Using device: {self.device}")

    def _validate_and_stream_data(self) -> Generator[SensorDataValidator, None, None]:
        """
        Private method to stream and validate data from the source file.
        This is a placeholder for actual data loading logic (e.g., from CSV, Parquet, live stream).
        """
        # In a real system, this would read from a file or a network stream.
        # Here, we simulate reading a large dataset.
        logging.info(f"Streaming and validating data from {self.config.file_path}...")
        # Placeholder: Simulate reading 1000 data points for demonstration
        for i in range(1000):
            mock_record = {
                "timestamp": 1672531200.0 + i,
                "sensor_id": f"RADAR_SWEEP_{i%4}",
                "sensor_type": "RADAR",
                "data": list(np.random.rand(1024)), # Simulate a 1D signal
                "metadata": {"azimuth": np.random.uniform(0, 360)}
            }
            try:
                validated_record = SensorDataValidator(**mock_record)
                yield validated_record
            except ValueError as e:
                logging.warning(f"Skipping invalid record: {mock_record}. Reason: {e}")
                continue

    def load_and_process_data(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Loads, validates, and processes sensor data, yielding batches of tensors.
        This method is memory-efficient, suitable for large datasets.

        Yields:
            Generator[Tuple[torch.Tensor, torch.Tensor], None, None]: A generator
            that yields batches of (features, labels). In this placeholder, labels
            are simulated.
        """
        batch_data = []
        batch_labels = [] # Labels would come from the data in a real scenario

        for record in self._validate_and_stream_data():
            batch_data.append(record.data)
            # Simulate a label (e.g., 0 for non-threat, 1 for threat)
            batch_labels.append(np.random.randint(0, 2))

            if len(batch_data) == self.config.batch_size:
                # Convert to tensor and move to the appropriate device
                features_tensor = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                # Add a channel dimension for compatibility with 1D ConvNets
                features_tensor = features_tensor.unsqueeze(1)

                yield features_tensor, labels_tensor

                # Clear batches
                batch_data = []
                batch_labels = []

        # Yield any remaining data that didn't form a full batch
        if batch_data:
            features_tensor = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
            features_tensor = features_tensor.unsqueeze(1)
            yield features_tensor, labels_tensor

if __name__ == '__main__':
    # Example usage
    # This demonstrates how the data loader would be used in a training script.

    # Create a dummy data file for the example
    import json
    dummy_file = "dummy_sensor_data.json"

    config = SensorDataLoaderConfig(file_path=dummy_file, batch_size=32)
    data_loader = SensorDataLoader(config)

    logging.info("Starting data loading process...")
    total_batches = 0
    for i, (features, labels) in enumerate(data_loader.load_and_process_data()):
        logging.info(f"Batch {i+1}: Features shape={features.shape}, Labels shape={labels.shape}")
        logging.info(f"  - Features device: {features.device}")
        total_batches += 1

    logging.info(f"Data loading complete. Total batches processed: {total_batches}")
