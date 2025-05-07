#!/usr/bin/env python3
import os

os.environ["DISPLAY"] = ":0"
# os.environ["XAUTHORITY"] = "/home/clover/.Xauthority" # Usually not needed if DISPLAY is set for same user
import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import threading  # Keep for signal handling
import signal
import sys
import time
import traceback
import socket  # For UDP streaming
import struct  # For packing data size

# --- Configuration ---
LAPTOP_IP = "100.111.15.64"  # Your laptop's IP

# Realsense settings (for capture and for sending via raw UDP)
RS_CAPTURE_WIDTH = 1920
RS_CAPTURE_HEIGHT = 1080
RS_CAPTURE_FPS = 30
RS_SEND_WIDTH = 1280  # Resolution to resize to before sending
RS_SEND_HEIGHT = 720
RS_UDP_PORT = 5003  # NEW Port for raw UDP Realsense stream
RS_JPEG_QUALITY = 40  # JPEG quality for Realsense

# Eye-Tracker Settings (for capture and for sending via raw UDP)
EYE_SEND_WIDTH = 640
EYE_SEND_HEIGHT = 480
EYE_CAPTURE_FPS = 30  # Target capture FPS
EYE_CAMERA_INDEX = 0
EYE_UDP_PORT = 5002
EYE_JPEG_QUALITY = 80  # JPEG quality for Eye tracker

# --- No GStreamer needed on Jetson for this approach ---
print("[INFO] GStreamer will NOT be used in this version of the script.")

# --- Global Variables ---
end_program = False
headless = False  # Set True to disable OpenCV POV window

# --- UDP Sockets ---
rs_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rs_target_addr = (LAPTOP_IP, RS_UDP_PORT)
print(f"[INFO] RealSense Stream UDP Target: {rs_target_addr}")

eye_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
eye_target_addr = (LAPTOP_IP, EYE_UDP_PORT)
print(f"[INFO] Eye Stream UDP Target: {eye_target_addr}")


def signal_handler(sig, frame):
    global end_program;
    print("\n[INFO] Ctrl+C pressed. Stopping program...");
    end_program = True


signal.signal(signal.SIGINT, signal_handler)


# --- Main Demo Function ---
def main():
    global end_program, headless, rs_sock, rs_target_addr, eye_sock, eye_target_addr

    rs_capture_pipeline = None
    eye_camera = None
    rs_capture_pipeline = rs.pipeline()

    try:
        # 1) RealSense camera config
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, RS_CAPTURE_WIDTH, RS_CAPTURE_HEIGHT, rs.format.bgr8, RS_CAPTURE_FPS)
        profile = rs_capture_pipeline.start(rs_config)
        print(f"[RealSense] Capture pipeline started ({RS_CAPTURE_WIDTH}x{RS_CAPTURE_HEIGHT} @ {RS_CAPTURE_FPS}fps)")
        print(f"[RealSense] Raw UDP stream will send frames resized to: {RS_SEND_WIDTH}x{RS_SEND_HEIGHT}")

        # 2) Eye camera config
        eye_camera = cv2.VideoCapture(EYE_CAMERA_INDEX)
        if not eye_camera.isOpened(): raise RuntimeError("Failed to open eye camera")

        # Try to set desired capture settings, but we resize anyway
        eye_camera.set(cv2.CAP_PROP_FRAME_WIDTH, EYE_SEND_WIDTH)  # Try to match send size
        eye_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, EYE_SEND_HEIGHT)
        eye_camera.set(cv2.CAP_PROP_FPS, EYE_CAPTURE_FPS)
        time.sleep(0.5)
        actual_cam_eye_w = int(eye_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_cam_eye_h = int(eye_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[USB EyeCam] Opened camera {EYE_CAMERA_INDEX}. Actual Read: {actual_cam_eye_w}x{actual_cam_eye_h}")
        print(f"[USB EyeCam] Raw UDP stream will send frames resized to: {EYE_SEND_WIDTH}x{EYE_SEND_HEIGHT}")

        # --- Create OpenCV Window for POV Display ---
        if not headless:
            cv2.namedWindow("RealSense POV", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("RealSense POV", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # --- No GStreamer pipelines to start ---
        print("[INFO] GStreamer pipelines are not used in this version.")

        # --- Main Loop ---
        print("[INFO] Starting capture loop. Press Ctrl+C to stop.")
        last_fps_time = time.time();
        frame_count = 0
        max_udp_payload = 65507

        while not end_program:
            # --- RealSense ---
            frameset = rs_capture_pipeline.poll_for_frames()
            if frameset:
                color_frame = frameset.get_color_frame()
                if color_frame:
                    color_bgr_rs_raw = np.asanyarray(color_frame.get_data())

                    # Display POV using OpenCV
                    if not headless:
                        rotated_image = cv2.rotate(color_bgr_rs_raw, cv2.ROTATE_180)
                        cv2.imshow("RealSense POV", rotated_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'): end_program = True; break

                    # Resize for sending
                    frame_rs_send = cv2.resize(color_bgr_rs_raw, (RS_SEND_WIDTH, RS_SEND_HEIGHT),
                                               interpolation=cv2.INTER_AREA)

                    # Compress and send
                    try:
                        ret_rs, buffer_rs = cv2.imencode('.jpg', frame_rs_send,
                                                         [int(cv2.IMWRITE_JPEG_QUALITY), RS_JPEG_QUALITY])
                        if ret_rs:
                            data_rs = buffer_rs.tobytes()
                            if len(data_rs) < max_udp_payload:
                                size_rs = len(data_rs)
                                packed_message_rs = struct.pack("Q", size_rs) + data_rs
                                bytes_sent = rs_sock.sendto(packed_message_rs, rs_target_addr)
                                print(f"[DEBUG JETSON RS] Sent {bytes_sent} bytes for RS frame.")  # DEBUG
                            else:
                                print(f"[WARN RS] RS Frame too large for UDP: {len(data_rs)}")
                        else:
                            print("[WARN RS] RS JPEG encoding failed")
                    except Exception as rs_e:
                        print(f"[ERROR RS Send Processing] {rs_e}")

            # --- Eye Camera (Raw UDP Stream) ---
            ret_eye_read, frame_eye_raw = eye_camera.read()
            if ret_eye_read and frame_eye_raw is not None:
                try:
                    frame_eye_send = cv2.resize(frame_eye_raw, (EYE_SEND_WIDTH, EYE_SEND_HEIGHT),
                                                interpolation=cv2.INTER_AREA)
                    ret_eye, buffer_eye = cv2.imencode('.jpg', frame_eye_send,
                                                       [int(cv2.IMWRITE_JPEG_QUALITY), EYE_JPEG_QUALITY])
                    if ret_eye:
                        data_eye = buffer_eye.tobytes()
                        if len(data_eye) < max_udp_payload:
                            size_eye = len(data_eye)
                            packed_message_eye = struct.pack("Q", size_eye) + data_eye
                            eye_sock.sendto(packed_message_eye, eye_target_addr)
                        # else: print(f"[WARN EyeCam] Frame too large for UDP: {len(data_eye)}")
                    # else: print("[WARN EyeCam] JPEG encoding failed")
                except Exception as eye_e:
                    print(f"[ERROR EyeCam Send] {eye_e}")

            # --- Loop Admin ---
            frame_count += 1;
            now = time.time()
            if now - last_fps_time >= 5.0:
                print(f"  FPS: {frame_count / (now - last_fps_time):.2f} (Overall loop)")
                frame_count = 0;
                last_fps_time = now
            if end_program: break
            if headless: time.sleep(0.001)  # Add small sleep if no cv2.waitKey

    except Exception as e:
        print(f"[ERROR] Exception: {e}"); traceback.print_exc()
    finally:  # --- Cleanup ---
        print("\n[INFO] Cleanup...");
        end_program = True
        # No GStreamer appsrc or pipelines to manage
        if rs_sock: rs_sock.close()
        if eye_sock: eye_sock.close()
        if 'rs_capture_pipeline' in locals() and rs_capture_pipeline: rs_capture_pipeline.stop()
        if eye_camera and eye_camera.isOpened(): eye_camera.release()
        if not headless: cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()