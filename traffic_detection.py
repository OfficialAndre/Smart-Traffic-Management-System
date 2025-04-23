import os
import io
import time
import cv2
import csv
import logging
import numpy as np
from flask import Flask, Response, render_template, jsonify, send_file, request
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
import mysql.connector
import mysql.connector.pooling

# Global variable declarations
db_connection = None
db_cursor = None

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# YOLO and DeepSORT model initialization
model = YOLO(r'C:\Users\andre\OneDrive\Documents\TrafficManagementSystem\yolov8n.pt')
tracker = DeepSort(max_age=30)

logging.basicConfig(filename='traffic_app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('traffic_app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Database connection pool (initialize ONCE)
cnx_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="my_pool",
    pool_size=5,
    host="localhost",
    user="root",
    password="newpassword123",
    database="traffic_management"
)

# Persistent vehicle counts
persistent_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0, "person": 0}
smoothing_window = 10
count_history = {key: deque(maxlen=smoothing_window) for key in persistent_counts.keys()}

# Color mapping
color_map = {
    "car": (0, 0, 255),
    "truck": (0, 255, 0),
    "bus": (255, 0, 0),
    "motorbike": (0, 255, 255),
    "person": (0, 0, 0)
}

# Insert vehicle data into the database
def insert_detection_data(vehicle_type, detected_time, color, density_value):
    try:
        db_connection = cnx_pool.get_connection()  # Get connection from pool
        db_cursor = db_connection.cursor()
        query = """
        INSERT INTO vehicle_data (type, detected_time, color, density_value)
        VALUES (%s, %s, %s, %s)
        """
        db_cursor.execute(query, (vehicle_type, detected_time, color, density_value))
        db_connection.commit()
        logger.info(f"Inserted {vehicle_type} at {detected_time} with density {density_value}")
    except mysql.connector.Error as err:
        logger.error(f"Database Error: {err}")
    finally:
        if db_connection and db_connection.is_connected():
            db_cursor.close()
            db_connection.close()  # Return to the pool by closing the connection

# Fetch filtered data from the database
@app.route("/filter_data", methods=["GET"])
def filter_data():
    try:
        db_connection = cnx_pool.get_connection()
        db_cursor = db_connection.cursor()

        vehicle_type = request.args.get("vehicle_type")
        selected_date = request.args.get("selected_date")
        end_date = request.args.get("end_date")

        query = "SELECT * FROM vehicle_data WHERE 1=1"
        params = []

        if vehicle_type:
            query += " AND type = %s"
            params.append(vehicle_type)
        if selected_date and end_date:
            query += " AND detected_time BETWEEN %s AND %s"
            params.append(selected_date)
            params.append(end_date)
        elif selected_date:
            query += " AND DATE(detected_time) = %s"
            params.append(selected_date)

        db_cursor.execute(query, params)
        results = db_cursor.fetchall()

        data = [
            {
                "ID": row[0],
                "type": row[1],
                "detected_time": row[2].strftime("%Y-%m-%d %H:%M:%S"),
                "color": row[3],
                "density_value": float(row[4]),
            }
            for row in results
        ]
        return jsonify(data)
    except mysql.connector.Error as err:
        logger.error(f"Database Error: {err}")
        return jsonify({"error": "Database error", "message": str(err)}), 500
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred", "message": str(e)}), 500
    finally:
        if db_connection and db_connection.is_connected():
            db_cursor.close()
            db_connection.close()

# Smoothing vehicle counts for accuracy
def smooth_counts(counts):
    for key in counts:
        count_history[key].append(counts[key])
        counts[key] = int(np.mean(count_history[key]))
    return counts

# Calculate vehicle density
def calculate_density(area_km2=1):
    active_vehicle_count = sum(1 for track in tracker.tracker.tracks if track.is_confirmed() and track.time_since_update <= 1)
    density = active_vehicle_count / area_km2 if area_km2 > 0 else 0
    return density

# Video streaming function for traffic monitoring
def video_stream():
    global persistent_counts
    cap = cv2.VideoCapture(r'C:/Users/andre/OneDrive/Documents/TrafficManagementSystem/video/traffic11.mp4')

    if not cap.isOpened():
        logger.error("Error: Unable to open video file")
        return

    logger.info("Video file opened successfully")

    frame_interval = 1
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("End of video stream or error reading frame.")
                break

            if frame_id % frame_interval == 0:
                results = model(frame, conf=0.3)
                detections = [
                    (int(detection.xyxy[0][0]), int(detection.xyxy[0][1]), int(detection.xyxy[0][2]), int(detection.xyxy[0][3]), float(detection.conf), int(detection.cls.item()))
                    for detection in results[0].boxes
                    if (model.names[int(detection.cls.item())]) in persistent_counts and float(detection.conf) > 0.3
                ]

                if detections:
                    bbox_xywh = [[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] for (x1, y1, x2, y2, _, _) in detections]
                    confidences = [conf for (_, _, _, _, conf, _) in detections]
                    classes = [cls_id for (_, _, _, _, _, cls_id) in detections]
                    formatted_detections = [(bbox_xywh[i], confidences[i], classes[i]) for i in range(len(detections))]

                    tracks = tracker.update_tracks(formatted_detections, frame=frame)

                    frame_vehicle_ids = set()  # Reset the set HERE
                    for track in tracks:
                        if track.is_confirmed() and track.time_since_update <= 1:
                            class_name = model.names[track.det_class]
                            if class_name in persistent_counts and track.track_id not in frame_vehicle_ids:
                                persistent_counts[class_name] += 1
                                frame_vehicle_ids.add(track.track_id)

                                detected_time = time.strftime('%Y-%m-%d %H:%M:%S')
                                density_value = calculate_density()
                                color = ','.join(map(str, color_map.get(class_name)))
                                insert_detection_data(class_name, detected_time, color, density_value)

                    persistent_counts = smooth_counts(persistent_counts)

                    for x1, y1, x2, y2, confidence, class_id in detections:
                        class_name = model.names[class_id]
                        color = color_map.get(class_name, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                ret, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            frame_id += 1

    except Exception as e:
        logger.exception("Exception in video_stream")
        print(f"Exception in video_stream: {e}")
    finally:
        if cap is not None:
            cap.release()
        logger.info("Video capture released.")

# Flask routes for video feed
@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Home page route
@app.route('/')
def index():
    density = calculate_density()
    total_vehicles = sum(persistent_counts.values())
    return render_template('index.html', vehicle_count=persistent_counts, density=density, total_vehicles=total_vehicles)

# Current count endpoint
@app.route('/current_count')
def current_count():
    density = calculate_density()
    total_vehicles = sum(persistent_counts.values())
    return jsonify(vehicle_count=persistent_counts, density=density, total_vehicles=total_vehicles)

# Download CSV file route
@app.route("/download_csv", methods=["GET"])
def download_csv():
    try:
        db_connection = cnx_pool.get_connection()
        db_cursor = db_connection.cursor()

        vehicle_type = request.args.get("vehicle_type")
        selected_date = request.args.get("selected_date")

        query = "SELECT * FROM vehicle_data WHERE 1=1"
        params = []

        if vehicle_type:
            query += " AND type = %s"
            params.append(vehicle_type)
        if selected_date:
            query += " AND DATE(detected_time) = %s"
            params.append(selected_date)

        db_cursor.execute(query, params)
        results = db_cursor.fetchall()

        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerow(["ID", "Type", "Detected Time", "Color", "Density Value"])

        for row in results:
            csv_writer.writerow([row[0], row[1], row[2], row[3], row[4]])

        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="filtered_traffic_data.csv"
        )

    except mysql.connector.Error as err:
        logger.error(f"Database Error: {err}")
        return jsonify({"error": "Database error", "message": str(err)}), 500

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred", "message": str(e)}), 500

    finally:
        if db_connection and db_connection.is_connected():
            db_cursor.close()
            db_connection.close()


# Close database connections after app context
@app.teardown_appcontext
def close_connection(exception):
    if hasattr(cnx_pool, 'close'):
        cnx_pool.close()
    else:
        try:
            for connection in cnx_pool._pool:
                if connection.is_connected():
                    connection.close()
        except AttributeError:
            pass

# Start Flask application
if __name__ == '__main__':
    app.run(debug=True)
