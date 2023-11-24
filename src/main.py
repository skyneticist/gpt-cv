import cv2
import time

from threading import Thread
from src.video_stream import VideoStream
from helpers import make_vision_request, save, update_message


def main():
    print("starting threaded video stream...")
    vs: VideoStream = VideoStream(1).start()
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    # fps global variable
    last_frame_time = 0

    # gpt-4 vision request globals
    processing = False
    message = ""

    def update_message(handle):
        nonlocal processing, message
        processing = False
        message = "Done"
        save(handle)

    while True:
        img = vs.read()
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - last_frame_time))
        last_frame_time = new_frame_time

        cv2.putText(img, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (100, 255, 0), 3, cv2.LINE_AA)

        (h, w) = img.shape[:2]
        frame_center_xy = (w//2, h//2)
        cv2.circle(img, frame_center_xy, 2, (255, 200, 1), -1)

        # Update message
        cv2.putText(img, message, (80, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("output", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("j") and not processing:
            processing = True
            message = "Context request received..."
            Thread(target=make_vision_request,
                   args=(img, update_message)).start()

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
