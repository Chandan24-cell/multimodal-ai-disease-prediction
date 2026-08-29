# backend/inference/dicom_loader.py
import pydicom
import numpy as np
from PIL import Image
import io
from typing import Union, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DICOMImageLoader:
    """
    Professional DICOM image loader for hospital PACS integration.
    Supports standard DICOM files from MRI, CT, X-Ray modalities.
    """
    
    @staticmethod
    def load_dicom_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load DICOM file and return image + metadata
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Dictionary containing:
                - image: PIL Image object (RGB, 224x224)
                - metadata: DICOM metadata dictionary
                - original_array: Original numpy array
        """
        try:
            # Read DICOM file
            ds = pydicom.dcmread(file_path)
            
            # Extract metadata
            metadata = DICOMImageLoader.extract_metadata(ds)
            
            # Convert pixel data to image
            image = DICOMImageLoader.pixel_array_to_image(ds)
            
            logger.info(f"Loaded DICOM: {metadata.get('modality', 'Unknown')} - {metadata.get('study_description', 'N/A')}")
            
            return {
                'image': image,
                'metadata': metadata,
                'original_array': ds.pixel_array if hasattr(ds, 'pixel_array') else None
            }
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_dicom_bytes(dicom_bytes: bytes) -> Dict[str, Any]:
        """
        Load DICOM from bytes (for file uploads)
        
        Args:
            dicom_bytes: Raw DICOM file bytes
            
        Returns:
            Dictionary with image and metadata
        """
        try:
            # Read from bytes
            ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
            
            # Extract metadata
            metadata = DICOMImageLoader.extract_metadata(ds)
            
            # Convert to image
            image = DICOMImageLoader.pixel_array_to_image(ds)
            
            return {
                'image': image,
                'metadata': metadata,
                'original_array': ds.pixel_array if hasattr(ds, 'pixel_array') else None
            }
            
        except Exception as e:
            logger.error(f"Error loading DICOM from bytes: {e}")
            raise
    
    @staticmethod
    def pixel_array_to_image(ds) -> Image.Image:
        """
        Convert DICOM pixel array to PIL Image suitable for ViT
        
        Handles:
        - MONOCHROME1 (inverse)
        - MONOCHROME2 (normal)
        - RGB color images
        - Grayscale to RGB conversion
        - Windowing for optimal contrast
        """
        
        # Get pixel array
        if 'PixelData' not in ds:
            raise ValueError("DICOM file has no pixel data")
        
        pixel_array = ds.pixel_array
        
        # Apply VOI LUT (Windowing) if available
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            try:
                window_center = ds.WindowCenter if isinstance(ds.WindowCenter, (int, float)) else ds.WindowCenter[0]
                window_width = ds.WindowWidth if isinstance(ds.WindowWidth, (int, float)) else ds.WindowWidth[0]
                
                min_val = window_center - window_width / 2
                max_val = window_center + window_width / 2
                
                pixel_array = np.clip(pixel_array, min_val, max_val)
                pixel_array = (pixel_array - min_val) / (max_val - min_val) * 255
                pixel_array = pixel_array.astype(np.uint8)
            except:
                # Fallback to simple normalization
                pass
        
        # Handle different photometric interpretations
        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                # Invert grayscale
                pixel_array = np.amax(pixel_array) - pixel_array
            elif ds.PhotometricInterpretation == 'MONOCHROME2':
                # Normal grayscale - no change needed
                pass
        
        # Normalize to 0-255 if needed
        if pixel_array.dtype != np.uint8:
            if pixel_array.max() != pixel_array.min():
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
                pixel_array = (pixel_array * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        
        # Handle 2D grayscale -> convert to RGB
        if len(pixel_array.shape) == 2:
            # Stack grayscale into 3 channels for ViT
            pixel_array = np.stack([pixel_array] * 3, axis=-1)
        
        # Handle 3D arrays (multiple slices) - take middle slice
        elif len(pixel_array.shape) == 3:
            if pixel_array.shape[0] > 1:
                # Take middle slice
                middle_idx = pixel_array.shape[0] // 2
                pixel_array = pixel_array[middle_idx]
                pixel_array = np.stack([pixel_array] * 3, axis=-1)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(pixel_array)
        
        # Resize to 224x224 for ViT (if needed)
        if pil_image.size != (224, 224):
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        return pil_image
    
    @staticmethod
    def extract_metadata(ds) -> Dict[str, Any]:
        """
        Extract relevant metadata from DICOM file for audit and display
        """
        def safe_get(attr, default=None):
            """Safely get DICOM attribute"""
            try:
                value = getattr(ds, attr)
                return str(value) if value is not None else default
            except:
                return default
        
        return {
            'patient_name': safe_get('PatientName'),
            'patient_id': safe_get('PatientID'),
            'patient_sex': safe_get('PatientSex'),
            'patient_age': safe_get('PatientAge'),
            'study_date': safe_get('StudyDate'),
            'study_time': safe_get('StudyTime'),
            'study_description': safe_get('StudyDescription'),
            'study_instance_uid': safe_get('StudyInstanceUID'),
            'series_description': safe_get('SeriesDescription'),
            'modality': safe_get('Modality'),
            'manufacturer': safe_get('Manufacturer'),
            'institution_name': safe_get('InstitutionName'),
            'sop_class_uid': safe_get('SOPClassUID'),
            'image_type': safe_get('ImageType'),
            'rows': safe_get('Rows'),
            'columns': safe_get('Columns'),
            'bits_allocated': safe_get('BitsAllocated'),
            'samples_per_pixel': safe_get('SamplesPerPixel'),
        }
    
    @staticmethod
    def convert_to_png_bytes(image: Image.Image) -> bytes:
        """
        Convert PIL image to PNG bytes for API response
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()