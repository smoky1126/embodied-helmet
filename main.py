#！/usr/bin/env python3
import os
import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import pyaudio
import wave
import threading
import signal
import sys
import time

# sys.path.append(os.path.abspath('/home/clover/GazeTracking'))
# from gaze_tracking import GazeTracking

import os
os.environ["DISPLAY"] = ":0"
os.environ["XAUTHORITY"] = "/home/clover/.Xauthority"

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

Gst.init(None)

headless = False
if not headless: 
    from pynput import mouse

# Global variables
recording = False
end_program = False
start_event = threading.Event()

def signal_handler(sig, frame):
    """Graceful shutdown on Ctrl+C."""
    global end_program
    print("[INFO] Ctrl+C pressed. Stopping program.")
    end_program = True

signal.signal(signal.SIGINT, signal_handler)

def on_click(x, y, button, pressed):
    global recording
    # If left mouse button is clicked, stop recording
    if button == mouse.Button.left and pressed:
        print(f"Left click detected at ({x}, {y})")
        recording = False  # Set recording to False to stop recording
        return False


def record_audio(output_audio_path, channels=1, rate=44100, chunk=1024):
    """
    Records audio until `recording` is False.
    Writes to WAV file once done.
    """
    global recording
    audio = pyaudio.PyAudio()
    print("[AUDIO] Audio thread started. Waiting to begin recording...")
    start_event.wait()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

    frames = []
    print("[AUDIO] Now recording...")
    try:
        while recording:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
    except Exception as e:
        print(f"[AUDIO] Error: {e}")
    finally:
        # Cleanup
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Write WAV file
        with wave.open(output_audio_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        print(f"[AUDIO] Finished. Wrote WAV to {output_audio_path}")


def main():
    global recording, end_program

    # 1) Create output folder with timestamp
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_directory = f'/home/clover/rs_recording_data/{current_time}'
    os.makedirs(output_directory, exist_ok=True)

    # 2) Define output paths
    output_video_path = os.path.join(output_directory, f"output_{current_time}.mp4")
    output_gaze_video_path = os.path.join(output_directory, f"gaze_output_{current_time}.mp4")
    output_audio_path = os.path.join(output_directory, f"audio_{current_time}.wav")
    print("[INFO] Output folder:", output_directory)

    # 3) GazeTracking init
    gaze = GazeTracking()

    # 4) RealSense camera config
    rs_resolution = (1920, 1080)
    pipeline_rs = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, rs_resolution[0], rs_resolution[1], rs.format.bgr8, 30)
    pipeline_rs.start(rs_config)
    print("[RealSense] Pipeline started.")

    # 5) Eye camera config
    gaze_resolution = (640, 480)
    eye_camera = cv2.VideoCapture(0)
    eye_camera.set(cv2.CAP_PROP_FRAME_WIDTH, gaze_resolution[0])
    eye_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, gaze_resolution[1])
    print("[USB-EyeTrackingCamera] Opened camera 0.")

    # 6) Build GStreamer pipeline (RealSense)
    #    Note: If you're on Jetson, keep "nvvidconv". Otherwise, replace with "videoconvert".
    rs_pipeline_launch = (
        "appsrc name=rs_appsrc is-live=true do-timestamp=true ! "
        "nvvidconv ! "  
        "x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast ! "
        "h264parse ! queue ! mp4mux ! "
        f"filesink location={output_video_path} sync=false"
    )
    gst_pipeline_rs = Gst.parse_launch(rs_pipeline_launch)
    rs_appsrc = gst_pipeline_rs.get_by_name("rs_appsrc")

    # Set caps on RealSense appsrc
    caps_rs = Gst.Caps.from_string(
        f"video/x-raw,format=I420,width={rs_resolution[0]},height={rs_resolution[1]},framerate=30/1"
    )
    rs_appsrc.set_property("caps", caps_rs)
    rs_appsrc.set_property("format", Gst.Format.TIME)  # use TIME-format
    rs_appsrc.set_property("stream-type", 0)  # 0=STREAM, 1=SEEKABLE, 2=RANDOM_ACCESS

    # 7) Build GStreamer pipeline (Eye camera)
    eye_pipeline_launch = (
        "appsrc name=eye_appsrc is-live=true do-timestamp=true ! "
        "nvvidconv ! " 
        "x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast ! "
        "h264parse ! queue ! mp4mux ! "
        f"filesink location={output_gaze_video_path} sync=false"
    )
    gst_pipeline_eye = Gst.parse_launch(eye_pipeline_launch)
    eye_appsrc = gst_pipeline_eye.get_by_name("eye_appsrc")

    # Set caps on Eye appsrc
    caps_eye = Gst.Caps.from_string(
        f"video/x-raw,format=I420,width={gaze_resolution[0]},height={gaze_resolution[1]},framerate=30/1"
    )
    eye_appsrc.set_property("caps", caps_eye)
    eye_appsrc.set_property("format", Gst.Format.TIME)
    eye_appsrc.set_property("stream-type", 0)

    # 8) Start GStreamer pipelines
    gst_pipeline_rs.set_state(Gst.State.PLAYING)
    gst_pipeline_eye.set_state(Gst.State.PLAYING)
    print("[GStreamer] Pipelines set to PLAYING.")

    # 9) Audio thread
    audio_thread = threading.Thread(target=record_audio, args=(output_audio_path,))
    audio_thread.start()

    # 10) display screen
    if not headless:
        cv2.namedWindow('RealSense', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('RealSense', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Count down time before recording
    if not headless:
        mouse_listener = mouse.Listener(on_click=on_click)
        mouse_listener.start()

    countdown = 3
    start_time = time.time()
    while time.time() - start_time < countdown:
        # Just show RealSense frames
        frameset = pipeline_rs.wait_for_frames()
        color_frame = frameset.get_color_frame()
        if color_frame:
            color_bgr = np.asanyarray(color_frame.get_data())
            remaining = countdown - int(time.time() - start_time)
            cv2.putText(color_bgr, f"Recording starts in {remaining} s",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            rotated_image = cv2.rotate(color_bgr, cv2.ROTATE_180)
            if not headless:
                cv2.imshow('RealSense', rotated_image)
            if cv2.waitKey(10) & 0xFF == ord('q') or end_program:
                end_program = True
                break

    # 11) Start actual recording
    recording = True
    start_event.set()  # Tells audio thread to start capturing
    print("[INFO] Recording now started. Press Q to stop.")

    try:
        while recording and not end_program:
            # --- RealSense ---
            frameset = pipeline_rs.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if color_frame:
                color_bgr = np.asanyarray(color_frame.get_data())

                # Convert BGR -> I420
                color_i420 = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2YUV_I420)
                color_bytes = color_i420.tobytes()

                # Create buffer
                gst_buffer_rs = Gst.Buffer.new_allocate(None, len(color_bytes), None)
                gst_buffer_rs.fill(0, color_bytes)

                # Optional manual PTS (if do-timestamp=true, GStreamer typically does it anyway)
                gst_buffer_rs.pts = int(time.time() * 1e9)
                gst_buffer_rs.duration = int(1e9/30)

                # Push
                ret_rs = rs_appsrc.emit("push-buffer", gst_buffer_rs)
                if ret_rs != Gst.FlowReturn.OK:
                    print(f"[RealSense] Push-buffer returned {ret_rs}")

                # Quick preview, rotate 180 degree to match small screen
                rotated_image = cv2.rotate(color_bgr, cv2.ROTATE_180)
                cv2.imshow('RealSense', rotated_image)

            # --- Eye Camera ---
            ret_eye, frame_eye = eye_camera.read()
            if ret_eye and frame_eye is not None:
                # GazeTracking
                gaze.refresh(frame_eye)
                # frame_eye = gaze.annotated_frame()
                # text = ""
                # if gaze.is_blinking():
                #     text = "Blinking" 
                # elif gaze.is_right():
                #     text = "Looking right"
                # elif gaze.is_left():
                #     text = "Looking left"
                # elif gaze.is_center():
                #     text = "Looking center"
                # else:
                #     text = "No gaze recognized"

                # cv2.putText(frame_eye, text, (10, 50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # left_pupil = gaze.pupil_left_coords()
                # right_pupil = gaze.pupil_right_coords()
                # cv2.putText(frame_eye, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
                # cv2.putText(frame_eye, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

                # Convert BGR -> I420
                eye_i420 = cv2.cvtColor(frame_eye, cv2.COLOR_BGR2YUV_I420)
                eye_bytes = eye_i420.tobytes()
                gst_buffer_eye = Gst.Buffer.new_allocate(None, len(eye_bytes), None)
                gst_buffer_eye.fill(0, eye_bytes)
                gst_buffer_eye.pts = int(time.time() * 1e9)
                gst_buffer_eye.duration = int(1e9/30)

                ret_eye_push = eye_appsrc.emit("push-buffer", gst_buffer_eye)
                if ret_eye_push != Gst.FlowReturn.OK:
                    print(f"[EyeCam] Push-buffer returned {ret_eye_push}")

            # Check for user wanting to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                end_program = True
                break

    except Exception as e:
        print(f"[MAIN] Exception: {e}")
    finally:
        recording = False
        print("[INFO] Stopping everything...")

        # 12) Send EOS so mp4mux finishes properly
        rs_appsrc.emit("end-of-stream")
        eye_appsrc.emit("end-of-stream")

        # Wait a moment to ensure mp4mux flushes
        time.sleep(1)

        # Set pipelines to NULL
        gst_pipeline_rs.set_state(Gst.State.NULL)
        gst_pipeline_eye.set_state(Gst.State.NULL)
        print("[GStreamer] Pipelines set to NULL. MP4 should be finalized.")

        # Stop RealSense
        pipeline_rs.stop()
        print("[RealSense] Pipeline stopped.")

        # Release camera
        eye_camera.release()
        if not headless:
            cv2.destroyAllWindows()

        # Join audio thread
        audio_thread.join()
        print(f"[INFO] Finished! Check your outputs in {output_directory}")


if __name__ == "__main__":
    main()
