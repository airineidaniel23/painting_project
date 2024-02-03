import cv2
import numpy as np

def load_and_preprocess_image(path):
    # Load the image
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

def find_keypoints_and_descriptors(image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors

def match_images(image1, image2):
    # Find keypoints and descriptors for both images
    keypoints1, descriptors1 = find_keypoints_and_descriptors(image1)
    keypoints2, descriptors2 = find_keypoints_and_descriptors(image2)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))
    matched_coords1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    matched_coords2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

    return matches, keypoints1, keypoints2, matched_coords1, matched_coords2

def main(image_path1, image_path2):
    # Load and preprocess images
    image1 = load_and_preprocess_image(image_path1)
    image2 = load_and_preprocess_image(image_path2)

    # Match images
    matches, keypoints1, keypoints2, mc1, mc2 = match_images(image1, image2)
    # Draw first 10 matches
    with open("matchhes.txt", "a") as file:
        for i in range(len(mc1)):
                file.write(f"{int(mc1[i][0])} {1080 - int(mc1[i][1])} {int(mc2[i][0])} {1080 - int(mc2[i][1])}\n")
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
main('set4/1.jpeg', 'set4/2.jpeg')
