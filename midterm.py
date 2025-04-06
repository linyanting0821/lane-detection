import numpy as np
import cv2

# 記錄過去的車道資訊
prev_s1, prev_b1, prev_s2, prev_b2 = None, None, None, None
done0 = 0
done1 = 0
done2 = 0

def do_img(img, idx):
    global prev_s1, prev_b1, prev_s2, prev_b2, done0, done1, done2

    ww, hh, rh, r = 640, 400, 0.6, 3
    xx1, yy1, xx2, yy2 = int(ww * 0.4), int(hh * rh), int(ww * 0.6), int(hh * rh)
    p1, p2, p3, p4 = [r, hh - r], [ww - r , hh - r], [xx2, yy2], [xx1, yy2]
    img1 = cv2.resize(img, (ww, hh))
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #ROI
    zero = np.zeros((hh, ww, 1), dtype="uint8")
    pts = np.array([p1, p2, p3, p4])
    zone = cv2.fillPoly(zero, [pts], 255)
    roi_img = cv2.bitwise_and(gray_img, zone)

    p1 = np.float32([p1, p2, p3, p4])
    p2 = np.float32([[0, 400], [640, 400], [640, 0], [0, 0]])
    m = cv2.getPerspectiveTransform(p1, p2)
    m_inv = cv2.getPerspectiveTransform(p2, p1)
    trans_img = cv2.warpPerspective(roi_img, m, (ww, hh))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    output = cv2.dilate(trans_img, kernel=kernel)
    output = cv2.GaussianBlur(output, (5,5), 0)
    output = cv2.erode(output, kernel=kernel)
    canny_img = cv2.Canny(output, 150, 200)
    canny_inv = cv2.warpPerspective(canny_img, m_inv, (ww, hh))
    #霍夫轉換
    HOUGH_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP = 20, 15, 70
    lines = cv2.HoughLinesP(canny_inv, 1, np.pi/180, HOUGH_THRESHOLD, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    done, s1, s2, b1, b2 = 0, None, None, None, None
    img2 = img1.copy()
    img3 = img1.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length < 40: 
                continue #過短線段不要
            if abs(x2 - x1) <= 5 or abs(y2 - y1) <= 5: 
                continue #過濾水平垂直線
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if min(x1, x2) < 100 or max(x1, x2) > ww - 130:
                continue #線段距離
            
            #限制斜率 左車線
            if -1.0 < slope < -0.4 and (s1 is None or slope < s1):
                if 100 < x1 < 300:
                    done |= 1
                    s1, b1 = slope, intercept
                    s1x1, s1y1, s1x2, s1y2 = x1, y1, x2, y2
            #右車線
            if 0.5 < slope < 2.0 and (s2 is None or slope > s2):
                if x2 < 475:
                    done |= 2
                    s2, b2 = slope, intercept
                    s2x1, s2y1, s2x2, s2y2 = x1, y1, x2, y2
    #兩條都沒找到
    if done == 0:
        done0 += 1
        # print(f"[{idx}] done = 0:使用上一幀資料")
        if prev_s1 is not None and prev_s2 is not None:
            s1, b1 = prev_s1, prev_b1
            s2, b2 = prev_s2, prev_b2
            done = 3
    #缺右
    elif done == 1:
        done1 += 1
        # print(f"[{idx}] done = 1:補右線")
        if prev_s2 is not None:
            s2, b2 = prev_s2, prev_b2
            done = 3
    #缺左
    elif done == 2:
        done2 += 1
        # print(f"[{idx}] done = 2:補左線")
        if prev_s1 is not None:
            s1, b1 = prev_s1, prev_b1
            done = 3

    if done == 3 and s1 is not None and s2 is not None:
        prev_s1, prev_b1 = s1, b1
        prev_s2, prev_b2 = s2, b2

        y1, y2 = hh - r, hh - hh*0.175
        x = int(((y1 - b1) / s1 + (y1 - b2) / s2) / 2)
        co = (255, 0, 0) if abs(x - int(ww/2)) <= 45 else (0, 0, 255)
        cv2.line(img2, (x, int(y1)), (x, int(y1 - 15)), co, 2)

        p1 = [int((y1 - b1) / s1), int(y1)]
        p2 = [int((y2 - b1) / s1), int(y2)]
        p3 = [int((y2 - b2) / s2), int(y2)]
        p4 = [int((y1 - b2) / s2), int(y1)]
        zone = np.zeros((hh, ww, 3), dtype='uint8')
        cv2.fillPoly(zone, [np.array([p1, p2, p3, p4])], (0, 50, 0))
        img2 = cv2.addWeighted(img2, 1.0, zone, 1.0, 0)
        print(f"第{idx}筆 | 左車道斜率: {s1:.3f}, 右車道斜率: {s2:.3f}")
        if 's1x1' in locals():
            cv2.line(img3, (s1x1, s1y1), (s1x2, s1y2), (255, 0, 0), 2)
        if 's2x1' in locals():
            cv2.line(img3, (s2x1, s2y1), (s2x2, s2y2), (0, 0, 255), 2)
        cv2.imshow("line", img3)

    x, y = int(ww / 2), hh - r
    cv2.line(img2, (x, y), (x, y - 12), (255, 0 , 0), 2)
    for i in range(1, 10):
        cv2.line(img2, (x - i * 15, y), (x - i * 15, y - 3), (0, 255, 0), 2)
        cv2.line(img2, (x + i * 15, y), (x + i * 15, y - 3), (0, 255, 0), 2)
    return img2

# 主程式
cap = cv2.VideoCapture('./lab-2/LaneVideo.mp4')
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, 20.0, (640, 400))
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame.")
        break
    processed = do_img(frame, idx)
    idx += 1
    cv2.imshow('oxxostudio', processed)
    out.write(processed)
    if cv2.waitKey(1) & 0xFF == 27:
        break

print(f"done0 = {done0}, done1 = {done1}, done2 = {done2}")
out.release()
cap.release()
cv2.destroyAllWindows()
