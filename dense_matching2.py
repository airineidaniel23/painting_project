import cv2
import numpy as np

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
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    # Extract the matched points
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Return pairs of matched points
    matched_points = list(zip(points1, points2))
    return matched_points

matched_points = match_points_sift('set4/1.jpeg', 'set4/2.jpeg')
print(len(matched_points))
with open("dense_matchesBunny.txt", "a") as file:
    for p in matched_points:
      file.write(f"{int(p[0][0])} {int(p[0][1])} {int(p[1][0])} {int(p[1][1])}\n")
