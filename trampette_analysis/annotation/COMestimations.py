import cv2
import numpy as np
import os
import glob

# Define segment names and mass percentages
segments = [
    ("Head", 8),
    ("Upper Arm (L)", 2.5),
    ("Upper Arm (R)", 2.5),
    ("Forearm + Hand (L)", 1.5),
    ("Forearm + Hand (R)", 1.5),
    ("Torso", 32),
    ("Thigh (L)", 10),
    ("Thigh (R)", 10),
    ("Shank + Foot (L)", 6),
    ("Shank + Foot (R)", 6)
]

# === Config ===
user_id = 15
folder_path = f"/Users/asmundur/Local/Documents/Master Thesis/Videos/{user_id}/TakeoffAndLandings"
image_extensions = ("*.jpg", "*.png", "*.jpeg")  # Supported formats
overwrite_original = False  # Set to False if you want to save as new files

def process_image(image_path):
    clicked_points = []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    original_image = image.copy()

    def redraw_image():
        """Redraw all points on the image."""
        nonlocal image
        image = original_image.copy()
        for i, (px, py) in enumerate(clicked_points):
            cv2.circle(image, (px, py), 3, (0, 0, 255), -1)
            cv2.putText(image, f"{i+1}", (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.imshow("Mark segments", image)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < len(segments):
                clicked_points.append((x, y))
                redraw_image()

    print(f"\nNow marking: {os.path.basename(image_path)}")
    print("Click on the following segments in order:")
    for i, (name, mass) in enumerate(segments):
        print(f"{i+1}. {name} ({mass}%)")
    print("Press ESC when done, or 'u' to undo last point.")

    cv2.imshow("Mark segments", image)
    cv2.setMouseCallback("Mark segments", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('u'):
            if clicked_points:
                clicked_points.pop()
                redraw_image()

    cv2.destroyAllWindows()

    if len(clicked_points) != len(segments):
        print("Not all segment points were marked. Skipping this image.")
        return

    # Compute COM
    total_mass = sum(mass for _, mass in segments)
    com_x = sum(x * mass for (x, y), (_, mass) in zip(clicked_points, segments)) / total_mass
    com_y = sum(y * mass for (x, y), (_, mass) in zip(clicked_points, segments)) / total_mass

    print(f"COM: x={com_x:.1f}, y={com_y:.1f}")

    # Draw COM
    output = original_image.copy()
    for i, (px, py) in enumerate(clicked_points):
        cv2.circle(output, (px, py), 3, (0, 0, 255), -1)
        cv2.putText(output, f"{i+1}", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.circle(output, (int(com_x), int(com_y)), 2, (0, 255, 0), -1)
    cv2.putText(output, "COM", (int(com_x)+6, int(com_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save
    if overwrite_original:
        cv2.imwrite(image_path, output)
    else:
        base, ext = os.path.splitext(image_path)
        new_path = f"{base}_with_COM{ext}"
        cv2.imwrite(new_path, output)

# === Run for all images ===
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))

if not image_files:
    print("No images found.")
else:
    for path in image_files:
        process_image(path)

print("\n✅ Done processing all images.")
