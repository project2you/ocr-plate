import os
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

# Base directory path (ปรับให้ตรงกับโครงสร้างไฟล์ของคุณ)
base_dir = "D:\\yolo_license_plate"
date_folder = "2024-11-23"  # วันที่ของข้อมูล
num_cameras = 10  # จำนวนกล้องทั้งหมด

# Sub-folder ภายใต้โฟลเดอร์แต่ละกล้อง
sub_folder = "plates"

# กำหนดขนาดขั้นต่ำและขั้นสูงของภาพ
MIN_WIDTH = 90  # ความกว้างขั้นต่ำ (พิกเซล)
MIN_HEIGHT = 60  # ความสูงขั้นต่ำ (พิกเซล)
MAX_WIDTH = 220  # ความกว้างขั้นสูง (พิกเซล)
MAX_HEIGHT = 160  # ความสูงขั้นสูง (พิกเซล)

# กำหนดโฟลเดอร์ output
output_folder_valid = os.path.join(base_dir, date_folder, "output_valid")
output_folder_invalid = os.path.join(base_dir, date_folder, "output_invalid")

# สร้างโฟลเดอร์สำหรับผลลัพธ์
os.makedirs(output_folder_valid, exist_ok=True)
os.makedirs(output_folder_invalid, exist_ok=True)

# รายงานผลการคัดกรอง
summary_report = defaultdict(lambda: {"valid": 0, "invalid": 0})


def filter_images_for_cameras(base_dir, date_folder, num_cameras, sub_folder):
    """
    ฟังก์ชันสำหรับคัดกรองภาพจากหลายกล้อง และรวมผลลัพธ์ไว้ในโฟลเดอร์เดียว
    """
    file_counter = 1  # ตัวนับไฟล์สำหรับการตั้งชื่อใหม่

    for camera_id in range(num_cameras):
        # สร้างเส้นทางโฟลเดอร์กล้อง
        input_folder = os.path.join(base_dir, date_folder, f"camera_{camera_id}", sub_folder)

        # ตรวจสอบว่าโฟลเดอร์ต้นทางมีอยู่หรือไม่
        if not os.path.exists(input_folder):
            print(f"Camera {camera_id}: Input folder does not exist: {input_folder}")
            continue

        print(f"Processing Camera {camera_id}: {input_folder}")

        # อ่านไฟล์ทั้งหมดในโฟลเดอร์
        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)

            # ตรวจสอบว่าไฟล์เป็นภาพหรือไม่
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image = cv2.imread(file_path)

                # ข้ามไฟล์ที่ไม่สามารถอ่านได้
                if image is None:
                    print(f"Skipped: {filename} - Not a valid image file.")
                    continue

                # ดึงขนาดของภาพ
                height, width = image.shape[:2]

                # ตรวจสอบว่าภาพมีขนาดที่เหมาะสมหรือไม่
                if MIN_WIDTH <= width <= MAX_WIDTH and MIN_HEIGHT <= height <= MAX_HEIGHT:
                    output_path = os.path.join(output_folder_valid, f"{file_counter:04d}.jpg")
                    cv2.imwrite(output_path, image)  # เก็บภาพที่ขนาดเหมาะสม
                    summary_report[camera_id]["valid"] += 1
                    print(f"Valid: {filename} - Size: {width}x{height}")
                else:
                    output_path = os.path.join(output_folder_invalid, f"{file_counter:04d}.jpg")
                    cv2.imwrite(output_path, image)  # เก็บภาพที่ขนาดไม่เหมาะสม
                    summary_report[camera_id]["invalid"] += 1
                    print(f"Invalid: {filename} - Size: {width}x{height}")

                file_counter += 1

    print(f"--- Processing completed for all cameras on {date_folder} ---")
    return summary_report


def generate_summary_report(summary_report, base_dir, date_folder):
    """
    สร้างรายงานผลการคัดกรอง และแสดงกราฟ
    """
    # สร้างไฟล์รายงาน
    report_path = os.path.join(base_dir, date_folder, "summary.txt")
    with open(report_path, "w") as f:
        f.write(f"Summary Report for {date_folder}\n")
        f.write("=" * 40 + "\n")
        for camera_id, results in summary_report.items():
            f.write(f"Camera {camera_id}:\n")
            f.write(f"  Valid Images: {results['valid']}\n")
            f.write(f"  Invalid Images: {results['invalid']}\n")
            f.write("-" * 20 + "\n")
    print(f"Summary report saved to: {report_path}")

    # สร้างกราฟแท่งสำหรับแต่ละกล้อง
    cameras = list(summary_report.keys())
    valid_counts = [summary_report[c]["valid"] for c in cameras]
    invalid_counts = [summary_report[c]["invalid"] for c in cameras]

    x = range(len(cameras))
    plt.bar(x, valid_counts, width=0.4, label="Valid", color="green", align="center")
    plt.bar(x, invalid_counts, width=0.4, label="Invalid", color="red", align="edge")
    plt.xlabel("Camera ID")
    plt.ylabel("Number of Images")
    plt.title(f"Image Categorization for {date_folder}")
    plt.xticks(x, [f"Camera {c}" for c in cameras], rotation=45)
    plt.legend()
    plt.tight_layout()

    # บันทึกกราฟแท่ง
    bar_graph_path = os.path.join(base_dir, date_folder, "summary_bar_graph.png")
    plt.savefig(bar_graph_path)
    print(f"Bar graph saved to: {bar_graph_path}")

    # แสดงกราฟแท่ง
    plt.show()

    # สร้างกราฟวงกลมสำหรับภาพรวม
    total_valid = sum(valid_counts)
    total_invalid = sum(invalid_counts)
    total_images = total_valid + total_invalid

    plt.figure(figsize=(6, 6))
    plt.pie(
        [total_valid, total_invalid],
        labels=["Valid", "Invalid"],
        autopct=lambda p: f'{p:.2f}%\n({int(p * total_images / 100)})',
        colors=["green", "red"],
        startangle=90
    )
    plt.title("Overall Image Categorization")
    plt.axis("equal")

    # บันทึกกราฟวงกลม
    pie_chart_path = os.path.join(base_dir, date_folder, "summary_pie_chart.png")
    plt.savefig(pie_chart_path)
    print(f"Pie chart saved to: {pie_chart_path}")

    # แสดงกราฟวงกลม
    plt.show()


# เรียกใช้งานฟังก์ชัน
summary_report = filter_images_for_cameras(base_dir, date_folder, num_cameras, sub_folder)
generate_summary_report(summary_report, base_dir, date_folder)
