import cv2

def main():
    # cap = cv2.VideoCapture("http://127.0.0.1:5000/video") 
    cap = cv2.VideoCapture(1)   # 0 = default camera

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Local Camera", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
