
import cv2
import numpy as np

canvas = []

def match_points_sift(image1_path, image2_path):
    # Read images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good_matches.append(m)

    # Extract the matched points
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Return pairs of matched points
    matched_points = list(zip(points1, points2))
    print(len(matched_points))
    return matched_points

def build_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = []
    for line in lines:
        a, b, x, y = map(float, line.split())

        row1 = [0, 0, 0, x, y, 1, -b*x, -b*y, -b]
        data.append(row1)

        row2 = [x, y, 1, 0, 0, 0, -a*x, -a*y, -a]
        data.append(row2)

    matrix = np.array(data)
    return matrix

def find_smallest_eigenvector(matrix):
    AtA = np.dot(matrix.T, matrix)

    eigenvalues, eigenvectors = np.linalg.eig(AtA)

    min_eigenvalue_index = np.argmin(eigenvalues)

    smallest_eigenvector = eigenvectors[:, min_eigenvalue_index]

    return smallest_eigenvector

def update_panorama(homography_matrix, image1_path, image2_path, output_path):
    # Load the images
    global canvas
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
    offsetx = width // 2
    offsety = height // 2
    translation_matrix = np.array([[1, 0, offsetx], [0, 1, offsety], [0, 0, 1]])
    adjusted_homography_matrix = np.dot(translation_matrix, homography_matrix)
    transformed_image2 = cv2.warpPerspective(image2, adjusted_homography_matrix, (width*2, height*2))

    # Create a canvas to fit both images
    if len(canvas) == 0:
        canvas_height = height * 2
        canvas_width = width * 2
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                canvas[i+offsety][j+offsetx] = image1[i][j]
    alpha = 0.5
    for i in range(transformed_image2.shape[0]):
        for j in range(transformed_image2.shape[1]):
            # if j < image1.shape[1] and i < image1.shape[0]:
            #     canvas[i + offsety, j + offsetx] = alpha * canvas[i + offsety, j + offsetx] + (1 - alpha) * transformed_image2[i, j]
            # elif np.any(transformed_image2[i, j] != 0):
            #     canvas[i + offsety, j + offsetx] = transformed_image2[i, j]
            if abs(sum(canvas[i, j]) - sum(transformed_image2[i, j])) > 60 and sum(canvas[i, j]) == 0:
                canvas[i, j] = transformed_image2[i, j]
            elif abs(sum(canvas[i, j]) - sum(transformed_image2[i, j])) > 60 and sum(transformed_image2[i,j]) == 0:
                pass
            else:
                canvas[i, j] = alpha * canvas[i, j] + (1 - alpha) * transformed_image2[i, j]
    # Save the result
    cv2.imwrite(output_path, canvas)
    print(f"Panorama saved as {output_path}")

def main():
    num_images = 2
    setNo = 1
    extension = "png"
    homography_matrices = []
    large_matrix = [[1,0,0],[0,1,0],[0,0,1]]

    for i in range(1, num_images):
        # Match points between image i and image i+1
        print("BUILDING MATCHES")
        matched_points = match_points_sift(f'set{setNo}/{i}.{extension}', f'set{setNo}/{i+1}.{extension}')

        # Write matched points to file
        with open("matches.txt", "w") as file:
            for p in matched_points:
                file.write(f"{int(p[0][0])} {int(p[0][1])} {int(p[1][0])} {int(p[1][1])}\n")

        # Build matrix and find homography
        print("BUILDING MATRIX")
        matrix = build_matrix("matches.txt")
        smallest_eigenvector = find_smallest_eigenvector(matrix)
        homography_matrix = smallest_eigenvector.reshape(3, 3)
        homography_matrices.append(homography_matrix)
        large_matrix = np.dot(large_matrix, homography_matrix)

        print("BUILDING PANORAMA")
        update_panorama(large_matrix, f'set{setNo}/{1}.{extension}', f'set{setNo}/{i+1}.{extension}', f'panorama_{1}_{i+1}.png')

if __name__ == "__main__":
    main()
