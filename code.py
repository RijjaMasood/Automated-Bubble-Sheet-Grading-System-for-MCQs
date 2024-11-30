import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return image, thresh

def detect_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = [c for c in contours if 1000 < cv2.contourArea(c) < 4000]
    return bubbles

def sort_bubbles_by_rows(bubbles, cols=4, y_tolerance=10):
    bubbles = sorted(bubbles, key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))
    rows = []
    current_row = []
    for bubble in bubbles:
        x, y, w, h = cv2.boundingRect(bubble)
        if not current_row:
            current_row.append(bubble)
        else:
            _, prev_y, _, prev_h = cv2.boundingRect(current_row[-1])
            if abs(y - prev_y) <= max(prev_h, h) + y_tolerance:
                current_row.append(bubble)
            else:
                rows.append(current_row)
                current_row = [bubble]
    if current_row:
        rows.append(current_row)
    sorted_rows = [sorted(row, key=lambda b: cv2.boundingRect(b)[0]) for row in rows]
    return sorted_rows

def is_filled(mask, bubble, filled_threshold_ratio=0.77, pixel_threshold=200):
    bubble_mask = np.zeros(mask.shape, dtype="uint8")
    cv2.drawContours(bubble_mask, [bubble], -1, 255, -1)
    roi = cv2.bitwise_and(mask, mask, mask=bubble_mask)
    _, roi_thresholded = cv2.threshold(roi, pixel_threshold, 255, cv2.THRESH_BINARY)
    bubble_pixels = cv2.countNonZero(roi_thresholded)
    x, y, w, h = cv2.boundingRect(bubble)
    bubble_area = w * h
    filled_ratio = bubble_pixels / bubble_area
    return filled_ratio > filled_threshold_ratio

def grade_bubble_sheet(grouped_bubbles, answer_key, original_image, thresh):
    scores = []
    marked_image = original_image.copy()
    correct_points = 1
    wrong_points = -0.25
    multi_mark_penalty = -0.5
    for i, row in enumerate(grouped_bubbles):
        filled = []
        for j, bubble in enumerate(row):
            if is_filled(thresh, bubble):
                filled.append(j)
        if len(filled) == 1:
            selected = filled[0]
            correct_answer = answer_key[i]
            if selected == correct_answer:
                scores.append(correct_points)
                cv2.drawContours(marked_image, [row[selected]], -1, (0, 255, 0), 3)
            else:
                scores.append(wrong_points)
                cv2.drawContours(marked_image, [row[selected]], -1, (0, 0, 255), 3)
        elif len(filled) > 1:
            scores.append(multi_mark_penalty)
            for f in filled:
                cv2.drawContours(marked_image, [row[f]], -1, (0, 0, 255), 3)
        else:
            scores.append(0)
    total_score = sum(scores)
    return marked_image, total_score

# Image Path
image_path = 'bubble_sheet.png'
answer_key = [0, 1, 1, 3, 0, 0, 2, 1, 3, 0]

# Preprocess Image
original_image, thresh = preprocess_image(image_path)
bubbles = detect_contours(thresh)
grouped_bubbles = sort_bubbles_by_rows(bubbles)
graded_image, score = grade_bubble_sheet(grouped_bubbles, answer_key, original_image, thresh)

print(f"Total Score: {score}")

# Fit window to screen
screen_width = 512 
screen_height = 512 
image_height, image_width = graded_image.shape[:2]
scale_width = screen_width / image_width
scale_height = screen_height / image_height
scale = min(scale_width, scale_height)
new_width = int(image_width * scale)
new_height = int(image_height * scale)
scaled_image = cv2.resize(graded_image, (new_width, new_height))

cv2.imshow("Graded Bubble Sheet", scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
