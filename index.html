
import cv2
import os
import urllib.request

# 얼굴 검출기 파일
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)

if not os.path.exists(CASCADE_FILE):
    print("얼굴 검출기 파일을 내려받는 중...")
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

if face_cascade.empty():
    print("얼굴 검출기를 불러올 수 없습니다.")
    raise SystemExit

# 카메라 실행
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    raise SystemExit

WIN = "Sheriff Robot View"
WARN_RATIO = 0.35

# XYZ 거리 추정 설정
FACE_WIDTH_CM = 16.0       # 얼굴의 실제 너비 가정값(cm)
FOCAL_LENGTH_PX = 600.0    # 카메라 초점거리의 픽셀값(보정 필요)

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # 화면 중심점
    center_x = W // 2
    center_y = H // 2

    # 화면 중심 표시
    cv2.drawMarker(
        img,
        (center_x, center_y),
        (255, 255, 0),
        cv2.MARKER_CROSS,
        20,
        2
    )

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2

        # Z: 얼굴 너비를 이용한 거리 추정(cm)
        Z = (FACE_WIDTH_CM * FOCAL_LENGTH_PX) / w

        # X, Y: 카메라 중심 기준 상대 좌표(cm)
        X = (cx - center_x) * Z / FOCAL_LENGTH_PX
        Y = (center_y - cy) * Z / FOCAL_LENGTH_PX

        # 얼굴이 가까우면 경고
        if w / W > WARN_RATIO:
            color = (0, 0, 255)
            status = "WARNING! TOO CLOSE!"
        else:
            color = (0, 255, 0)
            status = "TARGET LOCKED"

        # 얼굴 박스 및 십자선
        cv2.rectangle(
            img, (x, y), (x + w, y + h), color, 3
        )
        cv2.line(img, (cx - 20, cy), (cx + 20, cy), color, 2)
        cv2.line(img, (cx, cy - 20), (cx, cy + 20), color, 2)

        # 상태 표시
        ty = y - 10 if y > 30 else y + h + 25
        cv2.putText(
            img, status, (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )

        # XYZ 좌표 표시
        coord_text = f"X:{X:.1f} Y:{Y:.1f} Z:{Z:.1f} cm"

        coord_y = y + h + 25
        if coord_y > H - 10:
            coord_y = y - 35

        cv2.putText(
            img, coord_text, (x, coord_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 0), 2
        )

    # 화면 중심 좌표 기준 안내
    cv2.putText(
        img,
        "X: Right(+) / Left(-) | Y: Up(+) / Down(-) | Z: Distance",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow(WIN, img)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), 27):
        break

    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
