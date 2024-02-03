import cv2
import numpy as np

def create_panorama(homography_matrix, image1_path, image2_path, output_path):
    # Load the images
    image1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)

    # Check if images are loaded
    if image1 is None or image2 is None:
        print("Error loading images")
        return

    # Convert to RGB if images have an alpha channel
    if image1.shape[2] == 4:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGRA2BGR)
    if image2.shape[2] == 4:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)

    # Apply the homography transformation
    height, width = image2.shape[:2]
    transformed_image2 = cv2.warpPerspective(image2, homography_matrix, (width*3, height*3))

    # Create a canvas to fit both images
    canvas_height = max(image1.shape[0], transformed_image2.shape[0])
    canvas_width = image1.shape[1] + transformed_image2.shape[1]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Overlay both images at 50% opacity
    canvas[:image1.shape[0], :image1.shape[1]] = image1
    alpha = 0.5
    for i in range(transformed_image2.shape[0]):
        for j in range(transformed_image2.shape[1]):
            if j < image1.shape[1] and i < image1.shape[0]:
                canvas[i, j] = alpha * canvas[i, j] + (1 - alpha) * transformed_image2[i, j]
            elif np.any(transformed_image2[i, j] != 0):
                canvas[i, j] = transformed_image2[i, j]

    # Save the result
    cv2.imwrite(output_path, canvas)
    print(f"Panorama saved as {output_path}")

# Example usage
homography_matrix = np.array([[-0.004864,  0.000367, -0.202088],
 [-0.000345, -0.004861,  0.979331],
 [-0.,       -0.,       -0.004856]])
image1_path = '1.png'
image2_path = '2.png'
output_path = 'test.png'
create_panorama(homography_matrix, image1_path, image2_path, output_path)
