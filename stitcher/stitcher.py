import numpy as np
import cv2
import os
import re

def hsv2bgr(hsv):
    h, s, v = hsv[0], hsv[1]/100, hsv[2]/100
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
    return [int(b), int(g), int(r)]  # BGR format

def bgr2hsv(bgr):
    b, g, r = bgr[0]/255.0, bgr[1]/255.0, bgr[2]/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx * 100
    return [h, s, v]


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
    transformed_image2 = cv2.warpPerspective(image2, homography_matrix, (int(width*1), int(height*1)))

    # Create a canvas to fit both images
    canvas_height = max(image1.shape[0], transformed_image2.shape[0])
    canvas_width = int(image1.shape[1])
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Overlay both images at 50% opacity
    canvas[:image1.shape[0], :image1.shape[1]] = image1
    alpha = 0.5
    # nvs = 0
    # nvv = 0
    # nvi = 0
    # for i in range(transformed_image2.shape[0]):
    #     for j in range(transformed_image2.shape[1]):
    #         if np.sum(transformed_image2[i, j]) >= 10:
    #             nvi = nvi + 1
    #             nvs = bgr2hsv(image1[i,j])[1] + nvs
    #             nvv = bgr2hsv(image1[i,j])[2] + nvv
    # nvs = nvs/nvi
    # nvv = nvv/nvi

    for i in range(transformed_image2.shape[0]):
        for j in range(transformed_image2.shape[1]):
            if np.sum(transformed_image2[i, j]) >= 10:
                #pp = bgr2hsv(transformed_image2[i,j])
                #pp[1] = (nvs + pp[1]) / 2
                #pp[2] = (nvv + pp[2]) / 2
                #pp = hsv2bgr(pp)
                canvas[i, j] = transformed_image2[i,j]
            elif j < image1.shape[1] and i < image1.shape[0]:
                canvas[i,j] = image1[i,j]
            else:
                canvas[i,j] = [0,0,0]

    # Save the result
    cv2.imwrite(output_path, canvas)
    print(f"Panorama saved as {output_path}")

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



input_dir = 'frames'
output_dir = 'resframes'
painting_input = 'orig.png'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pattern = re.compile(r"frame_\d+\.jpg$")
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, precision=6)


for filename in os.listdir(input_dir):
    if pattern.match(filename):
        filename_matches = filename + ".txt"
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        input_matches_path = os.path.join(input_dir, filename_matches)
        matrix = build_matrix(input_matches_path)
        #print(matrix)
        smallest_eigenvector = find_smallest_eigenvector(matrix)
        #print("Smallest Eigenvector:")
        #print(smallest_eigenvector)
        print(np.dot(smallest_eigenvector.reshape(3, 3), np.array([649, 402, 1])))
        hm = smallest_eigenvector.reshape(3, 3)
        create_panorama(hm, input_path, painting_input, output_path)
        print("Created panorama for " + filename)