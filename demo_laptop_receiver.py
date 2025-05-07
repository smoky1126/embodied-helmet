import cv2
import numpy as np
import socket
import struct
import time
import threading
import traceback

# --- Configuration ---
LISTEN_IP = "0.0.0.0"
BUFFER_SIZE = 65536
SIZE_HEADER_LEN = struct.calcsize("Q")  # Should be 8

# Stream 1: RealSense
RS_UDP_PORT = 5003
RS_WINDOW_NAME = "RealSense Stream (UDP)"
RS_TARGET_WIDTH = 640  # For display window resizing
RS_TARGET_HEIGHT = 480

# Stream 2: Eye Tracker
EYE_UDP_PORT = 5002
EYE_WINDOW_NAME = "Eye Tracker Stream (UDP)"
EYE_TARGET_WIDTH = 640  # For display window resizing
EYE_TARGET_HEIGHT = 480
EYE_TEXT = "Eye tracking camera"
EYE_FONT = cv2.FONT_HERSHEY_SIMPLEX
EYE_FONT_SCALE = 1.0
EYE_FONT_COLOR = (0, 0, 255)
EYE_LINE_TYPE = 2

# --- Global Data Structures for Frames (Shared between threads) ---
# Use a simple approach: store the latest valid frame
latest_rs_frame = None
latest_eye_frame = None
receiver_running = True  # Flag to signal threads to stop


# --- Receiver Function (to be run in a thread) ---
def udp_receiver_thread(port, stream_name, target_width, target_height, is_eye_tracker=False):
    global latest_rs_frame, latest_eye_frame, receiver_running

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((LISTEN_IP, port))
        sock.settimeout(1.0)  # 1 second timeout
        print(f"[INFO] Thread for {stream_name} listening on {LISTEN_IP}:{port}")
    except socket.error as e:
        print(f"!!! Error binding socket for {stream_name} on port {port}: {e}")
        receiver_running = False  # Signal main thread and other threads to stop
        return

    frames_received_this_port = 0
    while receiver_running:
        try:
            data_received, addr = sock.recvfrom(BUFFER_SIZE)

            if len(data_received) < SIZE_HEADER_LEN:
                # print(f"[WARN {stream_name}] Packet too small for header.")
                continue

            expected_size = struct.unpack("Q", data_received[:SIZE_HEADER_LEN])[0]
            img_data_chunk = data_received[SIZE_HEADER_LEN:]
            img_data = img_data_chunk
            bytes_remaining = expected_size - len(img_data_chunk)

            while bytes_remaining > 0 and receiver_running:
                chunk, _ = sock.recvfrom(min(BUFFER_SIZE, bytes_remaining))
                if not chunk: img_data = None; break
                img_data += chunk
                bytes_remaining -= len(chunk)

            if not receiver_running: break  # Exit if global flag changed

            if img_data is None or len(img_data) != expected_size:
                # if img_data is not None: print(f"[WARN {stream_name}] Incomplete frame.")
                continue

            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                frames_received_this_port += 1
                # Resize if necessary (though Jetson should send at target size)
                # if frame.shape[1] != target_width or frame.shape[0] != target_height:
                #    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

                if is_eye_tracker:
                    cv2.putText(frame, EYE_TEXT, (10, frame.shape[0] - 10), EYE_FONT, EYE_FONT_SCALE, EYE_FONT_COLOR, EYE_LINE_TYPE)
                    latest_eye_frame = frame
                else:
                    latest_rs_frame = frame
                    RS_TEXT = "RealSense POV"
                    RS_FONT_COLOR = (0, 0, 255)  # Blue color in BGR
                    cv2.putText(frame, RS_TEXT, (10, frame.shape[0] - 10), EYE_FONT, 2.0, RS_FONT_COLOR, 3)
                    latest_rs_frame = frame
            # else: print(f"[WARN {stream_name}] Failed to decode JPEG.")

        except socket.timeout:
            continue  # Just means no data in the last second for this stream
        except struct.error as se:
            print(f"[ERROR {stream_name}] Struct unpack error: {se}")
            continue
        except Exception as e:
            print(f"[ERROR {stream_name}] Exception: {e}");
            traceback.print_exc()
            # receiver_running = False # Optionally stop all on critical error in one thread
            # break

    sock.close()
    print(f"[INFO] Thread for {stream_name} (port {port}) stopping. Total frames: {frames_received_this_port}")


# --- Main Part ---
if __name__ == "__main__":
    # Create OpenCV windows
    cv2.namedWindow(RS_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(RS_WINDOW_NAME, RS_TARGET_WIDTH, RS_TARGET_HEIGHT)
    cv2.namedWindow(EYE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(EYE_WINDOW_NAME, EYE_TARGET_WIDTH, EYE_TARGET_HEIGHT)

    # Create and start receiver threads
    rs_thread = threading.Thread(target=udp_receiver_thread,
                                 args=(RS_UDP_PORT, "RealSense", RS_TARGET_WIDTH, RS_TARGET_HEIGHT, False), daemon=True)
    eye_thread = threading.Thread(target=udp_receiver_thread,
                                  args=(EYE_UDP_PORT, "EyeTracker", EYE_TARGET_WIDTH, EYE_TARGET_HEIGHT, True),
                                  daemon=True)

    print("[INFO] Starting receiver threads...")
    rs_thread.start()
    eye_thread.start()

    print("[INFO] Displaying frames... Press 'q' in any window to quit.")

    try:
        while receiver_running:
            if latest_rs_frame is not None:
                cv2.imshow(RS_WINDOW_NAME, latest_rs_frame)

            if latest_eye_frame is not None:
                cv2.imshow(EYE_WINDOW_NAME, latest_eye_frame)

            key = cv2.waitKey(10)  # Refresh rate for display loop (e.g. 100 FPS)
            if key & 0xFF == ord('q'):
                print("Exit requested by user.")
                receiver_running = False  # Signal threads to stop
                break

            # Check if threads are still alive (optional, for more robust error handling)
            if not rs_thread.is_alive() and not eye_thread.is_alive() and receiver_running:
                print("[WARN] Both receiver threads have stopped unexpectedly.")
                break  # Exit main loop if threads died

            if not receiver_running:  # Check if a thread signalled to stop
                break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed in main display loop. Exiting.")
        receiver_running = False
    finally:
        receiver_running = False  # Ensure threads know to stop
        print("[INFO] Waiting for receiver threads to join...")
        if rs_thread.is_alive(): rs_thread.join(timeout=2)
        if eye_thread.is_alive(): eye_thread.join(timeout=2)

        cv2.destroyAllWindows()
        print("[INFO] All resources released.")