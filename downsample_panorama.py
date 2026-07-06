import cv2
import sys
import os

def downsample_image(
    input_path: str, 
    output_path: str = "panorama_128x128.jpg", 
    target_size: int = 128,
    preserve_aspect: bool = False
):
    """
    Load an image and downsample it to target_size x target_size (default 128x128).
    
    Parameters:
        input_path: Path to the input image file.
        output_path: Path to save the downsampled image.
        target_size: The target height and width (128).
        preserve_aspect: If True, pads the image to keep the aspect ratio.
                         If False, directly resizes (stretches/squishes) the image to a square.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist!")
        return False
        
    print(f"Loading '{input_path}'...")
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image '{input_path}'!")
        return False
        
    h, w = img.shape[:2]
    print(f"Original dimensions: {w}x{h} px")
    
    if preserve_aspect:
        # Resize preserving aspect ratio (letterbox padding)
        print("Downsampling with preserved aspect ratio...")
        scale = target_size / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        
        # Ensure dimensions are at least 1px
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # Resize to fit within target boundary
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Add black border padding to reach target_size x target_size
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        output_img = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        # Direct squish resize (stretches image to fit exactly target_size x target_size)
        print("Downsampling with direct squish resize...")
        output_img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
    # Save the output
    cv2.imwrite(output_path, output_img)
    print(f"Saved downsampled image -> {output_path} ({target_size}x{target_size} px)")
    return True

if __name__ == "__main__":
    # Use panorama_optimized.jpg as default if no file is provided
    input_file = "panorama_stitcher.jpg"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
    # Run direct squish resize
    downsample_image(input_file, "panorama_128x128.jpg", target_size=128, preserve_aspect=False)
    

