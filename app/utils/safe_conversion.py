# app/utils/safe_conversion.py
import numpy as np
import logging

def safe_float(value, default=0.0):
    """Safely convert any value to Python float"""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            elif value.size > 1:
                return float(value.flatten()[0])
            else:
                return default
        elif hasattr(value, 'item'):  # numpy scalar
            return float(value.item())
        elif hasattr(value, '__float__'):
            return float(value)
        else:
            return float(value)
    except (ValueError, TypeError, AttributeError):
        logging.getLogger(__name__).warning(f"Cannot convert {type(value)} to float: {value}")
        return default

def safe_int(value, default=0):
    """Safely convert any value to Python int"""
    try:
        return int(safe_float(value, default))
    except (ValueError, TypeError):
        return default

def safe_get_coordinates(keypoint_data, index=0):
    """Safely extract coordinates from keypoint data"""
    try:
        if isinstance(keypoint_data, (list, tuple)):
            if len(keypoint_data) > index:
                return safe_float(keypoint_data[index])
        elif isinstance(keypoint_data, np.ndarray):
            if keypoint_data.size > index:
                return safe_float(keypoint_data.flatten()[index])
        return 0.0
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error extracting coordinate {index}: {e}")
        return 0.0

def safe_array_access(array, index, default=0.0):
    """Safely access array element"""
    try:
        if hasattr(array, '__len__') and len(array) > index:
            return safe_float(array[index], default)
        elif hasattr(array, 'size') and array.size > index:
            return safe_float(array.flatten()[index], default)
        return default
    except Exception:
        return default