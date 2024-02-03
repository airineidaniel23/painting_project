import cv2
import numpy as np
import os
import re

orig_refs = []
expected_refs = 15

def sort_in_groups_of_three_by_y(lst):
    sorted_list = []
    for i in range(0, len(lst), 3):
        chunk = lst[i:i+3]
        sorted_chunk = sorted(chunk, key=lambda x: x[1])
        sorted_list.extend(sorted_chunk)
    return sorted_list

def process_orig(image_path):
    original_image = cv2.imread(image_path)
    
    height, width = original_image.shape[:2]
    i = 0
    image = original_image.copy()
    image = cv2.medianBlur(image, 5)  # Using a kernel size of 5
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range for green color in HSV
    red_lower = np.array([0, 100, 60])  # Adjust these values as needed
    red_upper = np.array([20, 255, 255])
    
    red2_lower = np.array([170, 100, 60])  # Adjust these values as needed
    red2_upper = np.array([180, 255, 255])


    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    mask_red2 = cv2.inRange(hsv, red2_lower, red2_upper)

    mask_combined = cv2.bitwise_or(mask_red, mask_red2)
    
    filtered_image = cv2.bitwise_and(image, image, mask=mask_combined)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    # Apply median filter to reduce image noise
    gray = cv2.medianBlur(gray, 3)  # Using a kernel size of 5
    gray = cv2.medianBlur(gray, 5)  # Using a kernel size of 5
    cv2.imwrite('origmask.jpg', gray)

    # Detect edges using Canny
    edged = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calculate the center of each contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centerX = int(M["m10"] / M["m00"])
            centerY = int(M["m01"] / M["m00"])
            if (centerY > height * 0.7 ):
                continue

            #print(f"upper left Center at ({centerX}, {centerY}) is good.")
            i = i + 1
            # Mark the center with a green dot for visualization
            cv2.circle(image, (centerX, centerY), 2, (0, 255, 0), -1)
            orig_refs.append([centerX, centerY])

    # Save the image with centers marked (if they are black)
    cv2.imwrite("orig_detected.jpg", image)
    print(f"Found {i} centers")

    # Display the result
    #cv2.imshow('Centers Marked on Filtered Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if i == expected_refs:
        return 0
    else:
        return 1

def process_image(image_path, output_path):
    global orig_refs
    original_image = cv2.imread(image_path)
    temp_refs = []
    
    height, width = original_image.shape[:2]
    i = 0
    image = original_image.copy()
    image = cv2.medianBlur(image, 5)  # Using a kernel size of 5
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range for green color in HSV
    red_lower = np.array([0, 100, 60])  # Adjust these values as needed
    red_upper = np.array([20, 255, 255])
    
    red2_lower = np.array([170, 100, 60])  # Adjust these values as needed
    red2_upper = np.array([180, 255, 255])


    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    mask_red2 = cv2.inRange(hsv, red2_lower, red2_upper)

    mask_combined = cv2.bitwise_or(mask_red, mask_red2)
    
    filtered_image = cv2.bitwise_and(image, image, mask=mask_combined)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    # Apply median filter to reduce image noise
    gray = cv2.medianBlur(gray, 3)  # Using a kernel size of 5
    gray = cv2.medianBlur(gray, 5)  # Using a kernel size of 5
    cv2.imwrite('maskred.jpg', gray)

    # Detect edges using Canny
    edged = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calculate the center of each contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centerX = int(M["m10"] / M["m00"])
            centerY = int(M["m01"] / M["m00"])
            if (centerY > height * 0.5 ):
                continue

            #print(f"upper left Center at ({centerX}, {centerY}) is good.")
            i = i + 1
            # Mark the center with a green dot for visualization
            cv2.circle(image, (centerX, centerY), 2, (0, 255, 0), -1)
            temp_refs.append([centerX, centerY])

    print(f"Found {i} centers")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    orig_refs = sorted(orig_refs, key=lambda x: x[0])
    orig_refs = sort_in_groups_of_three_by_y(orig_refs)
    temp_refs = sorted(temp_refs, key=lambda x: x[0])
    temp_refs = sort_in_groups_of_three_by_y(temp_refs)
    if i == expected_refs:
        cv2.imwrite(output_path, image)
        with open(output_path + ".txt", "a") as file:
            for j in range(0,expected_refs):
                file.write(f"{temp_refs[j][0]} {temp_refs[j][1]} {orig_refs[j][0]} {orig_refs[j][1]}\n")
    if i == expected_refs:
        return 0
    else:
        return 1

# Directory paths
input_dir = 'vid'
output_dir = 'resultvid'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Regex to match files like 'frame1.jpg', 'frame2.jpg', etc.
pattern = re.compile(r"frame_\d+\.jpg")

# Process each file in the directory

j = 0
k = 0
process_orig("orig.jpg")
for filename in os.listdir(input_dir):
    if pattern.match(filename):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        j = j + process_image(input_path, output_path)
        print(f"Processed {filename}")
        k = k + 1

print(f"We had {j} bad apples out of {k} => {round((float(j)/float(k) * 100),2)}%")