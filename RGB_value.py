import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_rgb_values_in_chunks(image_path, chunk_size=1000):
    """
    Extract RGB values from an orthophoto image using chunk processing to handle large images.
    
    Args:
        image_path (str): Path to the orthophoto image
        chunk_size (int): Number of rows to process at once
        
    Returns:
        pandas.DataFrame: DataFrame containing pixel coordinates and RGB values
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width = img_rgb.shape[:2]
    
    # Create output file path
    output_path = Path(image_path).with_suffix('.csv')
    
    # Process image in chunks to save memory
    print(f"Processing image of size {width}x{height} pixels")
    
    # Write header to CSV
    pd.DataFrame(columns=['x', 'y', 'red', 'green', 'blue']).to_csv(output_path, index=False)
    
    # Process chunks
    for start_row in tqdm(range(0, height, chunk_size)):
        end_row = min(start_row + chunk_size, height)
        
        # Extract chunk
        chunk = img_rgb[start_row:end_row, :, :]
        
        # Create coordinate arrays for this chunk
        y_coords, x_coords = np.meshgrid(
            np.arange(start_row, end_row),
            np.arange(width),
            indexing='ij'
        )
        
        # Create DataFrame for this chunk
        chunk_df = pd.DataFrame({
            'x': x_coords.flatten(),
            'y': y_coords.flatten(),
            'red': chunk[:, :, 0].flatten(),
            'green': chunk[:, :, 1].flatten(),
            'blue': chunk[:, :, 2].flatten()
        })
        
        # Append to CSV without loading entire file into memory
        chunk_df.to_csv(output_path, mode='a', header=False, index=False)
        
    return output_path

def analyze_rgb_statistics_in_chunks(csv_path, chunk_size=10000):
    """
    Calculate basic statistics of RGB values by reading CSV in chunks.
    
    Args:
        csv_path (str): Path to the CSV file containing RGB values
        chunk_size (int): Number of rows to process at once
        
    Returns:
        dict: Dictionary containing RGB statistics
    """
    # Initialize variables for running calculations
    n = 0  # count
    rgb_sum = np.zeros(3)
    rgb_sq_sum = np.zeros(3)
    rgb_min = np.array([255, 255, 255])
    rgb_max = np.zeros(3)
    
    print("Calculating statistics...")
    
    # Process CSV in chunks
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size)):
        rgb_values = chunk[['red', 'green', 'blue']].values
        
        # Update running calculations
        n += len(rgb_values)
        rgb_sum += rgb_values.sum(axis=0)
        rgb_sq_sum += (rgb_values ** 2).sum(axis=0)
        rgb_min = np.minimum(rgb_min, rgb_values.min(axis=0))
        rgb_max = np.maximum(rgb_max, rgb_values.max(axis=0))
    
    # Calculate final statistics
    rgb_mean = rgb_sum / n
    rgb_std = np.sqrt((rgb_sq_sum / n) - (rgb_mean ** 2))
    
    return {
        'red_mean': rgb_mean[0],
        'green_mean': rgb_mean[1],
        'blue_mean': rgb_mean[2],
        'red_std': rgb_std[0],
        'green_std': rgb_std[1],
        'blue_std': rgb_std[2],
        'red_min': rgb_min[0],
        'green_min': rgb_min[1],
        'blue_min': rgb_min[2],
        'red_max': rgb_max[0],
        'green_max': rgb_max[1],
        'blue_max': rgb_max[2]
    }

def main():
    # Replace with your image path
    image_path = "./20230707_HSL_Color.png"
    
    try:
        # Extract RGB values
        print("Extracting RGB values...")
        output_path = extract_rgb_values_in_chunks(image_path)
        print(f"RGB values saved to: {output_path}")
        
        # Calculate and display statistics
        stats = analyze_rgb_statistics_in_chunks(output_path)
        print("\nRGB Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()