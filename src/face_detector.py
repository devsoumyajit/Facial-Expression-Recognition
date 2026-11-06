import os
import sys
import cv2


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "src",
    "haarcascade_frontalface_default.xml"
)


def load_face_detector():
    print("LOADING FACE DETECTOR")

    if not os.path.exists(CASCADE_PATH):
        print("Haar Cascade file not found:")
        print(CASCADE_PATH)
        sys.exit(1)

    detector = cv2.CascadeClassifier(CASCADE_PATH)

    if detector.empty():
        print("Failed to load Haar Cascade.")
        sys.exit(1)

    print("Face detector loaded successfully.")

    return detector


def detect_faces(detector, image_path):
    print("\nLOADING IMAGE")

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image:")
        print(image_path)
        sys.exit(1)

    print(f"Original image shape: {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"\nFaces detected: {len(faces)}")

    for i, (x, y, w, h) in enumerate(faces, start=1):
        print(f"Face {i}: x={x}, y={y}, width={w}, height={h}")

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "face_detection_result.jpg"
    )

    cv2.imwrite(output_path, image)

    print("\nDetection result saved to:")
    print(output_path)


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python src/face_detector.py path_to_image")

        print("\nExample:")
        print("python src/face_detector.py test.jpg")

        sys.exit(1)

    image_path = sys.argv[1]

    detector = load_face_detector()

    detect_faces(detector, image_path)

    print("\nFACE DETECTION COMPLETED")


if __name__ == "__main__":
    main()