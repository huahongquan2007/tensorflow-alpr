import json
import cv2
import os
from collections import namedtuple


Point = namedtuple('Point', ['x', 'y'])


class Label:
    def __init__(self, p1=None, p2=None, p3=None, p4=None, name=None, path=None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.name = name
        self.path = path

    def import_data(self, data):
        self.name = data["name"]
        self.path = data["path"]
        self.p1 = Point(data["p1"]["x"], data["p1"]["y"])
        self.p2 = Point(data["p2"]["x"], data["p2"]["y"])
        self.p3 = Point(data["p3"]["x"], data["p3"]["y"])
        self.p4 = Point(data["p4"]["x"], data["p4"]["y"])

    def export_data(self):
        return {
            "name": self.name,
            "path": self.path,
            "p1": self.p1._asdict(),
            "p2": self.p2._asdict(),
            "p3": self.p3._asdict(),
            "p4": self.p4._asdict(),
        }

points = []
image = None
MARKER_SIZE = 5


def mouse_callback(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONUP:
        p = Point(x=x, y=y)
        if p not in points:
            points.append(Point(x=x, y=y))
            print("append", p)
        cv2.circle(image, (x, y), MARKER_SIZE, (0, 255, 0), -1)
        cv2.imshow('image', image)


cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)


def write_label(label, file_name):
    data = label.export_data()
    with open(file_name, 'w') as output_file:
        output_file.write(json.dumps(data))


def read_label(file_name):
    with open(file_name, 'r') as input_file:
        data_str = input_file.readline()
        data = json.loads(data_str)
        label = Label()
        label.import_data(data)
        return label


def label_image(img_path, out_path):
    global image, points
    print("img_path", img_path)
    points = []
    image = cv2.imread(img_path)

    if os.path.exists(out_path):
        label = read_label(out_path)
        points = [label.p1, label.p2, label.p3, label.p4]
        cv2.putText(image, label.name, (10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 255, 0), thickness=10, lineType=cv2.LINE_AA)
        cv2.circle(image, label.p1, MARKER_SIZE, (0, 255, 0), -1)
        cv2.circle(image, label.p2, MARKER_SIZE, (0, 255, 0), -1)
        cv2.circle(image, label.p3, MARKER_SIZE, (0, 255, 0), -1)
        cv2.circle(image, label.p4, MARKER_SIZE, (0, 255, 0), -1)

    cv2.imshow('image', image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("reset")
            points = []
            image = cv2.imread(img_path)
            cv2.imshow('image', image)
        elif key == ord('n'):
            print("next")
            return True
        elif key == ord('q'):
            return False
        elif key == ord('s'):
            print("save")
            if len(points) == 4:
                break
            else:
                print("not enough points", points)
    name = input("Please type the license number: ")
    if len(name) > 0:
        print("name", name)
        label = Label(points[0], points[1], points[2], points[3], name, img_path)
        write_label(label, out_path)
    else:
        print("name is too short")
    return True


if __name__ == "__main__":
    data_dir = "data/toll-plaza-a/raw/"
    list_files = [ os.path.join(data_dir, name) for name in os.listdir(data_dir)]
    for name in list_files:
        out_name = name.replace(".png", ".json").replace("/raw/", "/labels/")
        result = label_image(name, out_name)
        if not result:
            break
