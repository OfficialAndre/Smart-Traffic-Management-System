# Smart Traffic Management System 🚦📊

An AI-powered traffic monitoring and analytics system built using **YOLOv8**, **OpenCV**, and **Flask**. This project detects and classifies vehicles in real-time, calculates traffic density, and displays a live dashboard with actionable insights for smarter traffic control.

---

## 🔧 Features

- 🚗 Vehicle detection and classification (e.g., cars, trucks, buses)
- 📈 Real-time traffic density and vehicle count tracking
- 🎨 Color-coded visualizations per vehicle type
- 🧠 YOLOv8 integration for object detection
- 🌐 Flask web dashboard for monitoring
- 🔁 Optional video looping or live camera feed
- ✅ Non-redundant counting logic to avoid double counts

---

## 📦 Technologies Used

- Python 3.x
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Flask
- NumPy

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/OfficialAndre/Smart-Traffic-Management-System.git
cd Smart-Traffic-Management-System

Set Up a Virtual Environment (Optional but recommended)
python -m venv venv
venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt

Project Structure
├── traffic_detection.py          # Main Python script
├── coco8.yaml                    # Custom class configuration for YOLO
├── templates/
│   └── dashboard.html            # Flask dashboard UI
├── static/
│   └── css, js, etc.             # Styling and assets
├── requirements.txt              # Dependencies
└── README.md                     # Project overview

🧠 Future Enhancements
Live camera feed support (e.g., IP/RTSP)

Historical traffic data logging

Interactive charts (e.g., traffic patterns over time)

Integration with traffic lights and control systems

👤 Author
Andre F. McLean
📧 andrefmclean@gmail.com
📱 1-876-853-0533 | 1-208-695-5219

📘 License
This project is open-source and free to use under the MIT License.


---

Let me know if you want to add screenshots, a demo video, or links to related projects! I can also help you generate a badge-style header with stats (stars, forks, etc.) if you’re interested in making it flashier.

