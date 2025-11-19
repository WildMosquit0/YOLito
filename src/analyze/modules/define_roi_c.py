import cv2
import pandas as pd
import numpy as np
import os

# Global variables for interactive ROI selection
center = None

image = None
clone = None
selected_rois = {}  # Dictionary to store ROIs per image

def draw_circle(event, x, y, flags, param):
    """ Mouse callback function to select the center of the circle. """
    global center, image, clone, radius

    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        image = clone.copy()
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.imshow("Select Circle ROI", image)

def update_radius(value):
    """ Updates the radius based on trackbar position. """
    global radius, image, clone, center
    radius = max(5, value)  # Ensure minimum radius
    if center:
        image = clone.copy()
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.imshow("Select Circle ROI", image)

def crop_circle(img, center, radius):
    """ Extracts a circular region from the image. """
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    
    # Apply mask and extract circular region
    result = cv2.bitwise_and(img, mask)
    
    # Create bounding box for circular crop
    x, y = center
    x1, x2 = max(0, x - radius), min(img.shape[1], x + radius)
    y1, y2 = max(0, y - radius), min(img.shape[0], y + radius)
    
    cropped = result[y1:y2, x1:x2]
    return cropped

def filter_csv(csv_path, selected_rois):
    """ Filters CSV points inside the defined circular ROIs per image. """
    df = pd.read_csv(csv_path)

    # Ensure CSV contains required columns
    if not {"image_name", "x", "y"}.issubset(df.columns):
        raise ValueError("CSV file must contain 'image_name', 'x', and 'y' columns.")

    # Convert columns to expected types
    df["image_name"] = df["image_name"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    filtered_dfs = []
    
    for image_name, (center, radius) in selected_rois.items():
        # Remove extension before filtering (e.g., control_1.jpg → control_1)
        base_name = os.path.splitext(image_name)[0]
        base_name = base_name.split("_middle_frame")[0]
        img_df = df[df["image_name"].str.startswith(base_name)].copy()
        
        if img_df.empty:
            print(f"⚠️ Warning: No matching rows for image {image_name} in CSV.")
            continue

        img_df["distance"] = np.sqrt((img_df["x"] - center[0]) ** 2 + (img_df["y"] - center[1]) ** 2)
        
        # Keep only points within the circle
        img_filtered = img_df[img_df["distance"] <= radius].drop(columns=["distance"])
        
        if img_filtered.empty:
            print(f"⚠️ No points inside the circle for {image_name}.")
        
        filtered_dfs.append(img_filtered)

    # Concatenate all filtered data
    final_filtered_df = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else pd.DataFrame()
    
    return final_filtered_df

def process_images(image_dir, csv_path, output_dir):
    global image, clone, center, radius, selected_rois

    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Error loading {img_file}, skipping...")
            continue

        clone = image.copy()
        center = None  # Reset center for each image

        cv2.imshow("Select Circle ROI", image)
        cv2.setMouseCallback("Select Circle ROI", draw_circle)
        cv2.createTrackbar("Radius", "Select Circle ROI", radius, 600, update_radius)

        print(f"🟢 Processing {img_file}: Click to set the center, adjust radius, then press 'c' to confirm.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and center:
                break

        cv2.destroyAllWindows()

        # Store selected ROI for filtering
        selected_rois[img_file] = (center, radius)

        # Crop circular region and save
        cropped = crop_circle(image, center, radius)
        cropped_path = os.path.join(output_dir, f"cropped_{img_file}")
        cv2.imwrite(cropped_path, cropped)
        print(f"✅ Saved cropped image: {cropped_path}")

    # Process CSV file
    filtered_data = filter_csv(csv_path, selected_rois)
    
    if filtered_data.empty:
        print("⚠️ No points matched any of the selected ROIs. Check filenames or coordinates.")
    else:
        filtered_csv_path = os.path.join(output_dir, "filtered_data.csv")
        filtered_data.to_csv(filtered_csv_path, index=False)
        print(f"✅ Filtered CSV saved: {filtered_csv_path}")

if __name__ == "__main__":
    radius = 300  # Default radius
    image_dir = "/home/bohbot/workspace/projects/arad/r/frames"  # Change to your image directory
    csv_path = "/home/bohbot/workspace/projects/arad/r/results.csv"  # Change to your CSV file
    output_dir = "/home/bohbot/workspace/projects/arad/r"  # Directory to save cropped images and filtered CSV
    process_images(image_dir, csv_path, output_dir)
