import cv2

def subtract_background(image_path1, image_path2, output_path):
    # Load the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    # Perform the subtraction
    # Note: cv2.subtract will clip values at 0, preventing negative values
    result = cv2.absdiff(img2, img1)
    
    # Save the resulting image
    cv2.imwrite(output_path, result)
    print(f"Result saved to {output_path}")

# Replace '1.jpg' and '2.jpg' with the correct paths to your images
subtract_background('1.jpg', '2.jpg', 'result.jpg')
