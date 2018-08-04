# -*- coding: utf-8 -*-
# 摄像头头像识别
import face_recognition
import cv2
import ft2

source = "rtsp://admin:5417010101xx@192.168.1.61/Streaming/Channels/1"
cam = cv2.VideoCapture(source)

# 本地图像
zwh_image = face_recognition.load_image_file("zwh.jpg")
zwh_face_encoding = face_recognition.face_encodings(zwh_image)[0]

# 本地图像二
chenduling_image = face_recognition.load_image_file("chenduling.jpg")
chenduling_face_encoding = face_recognition.face_encodings(chenduling_image)[0]

# 本地图片三
liujunbo_image = face_recognition.load_image_file("liujunbo.jpg")
liujunbo_face_encoding = face_recognition.face_encodings(liujunbo_image)[0]

# Create arrays of known face encodings and their names
# 脸部特征数据的集合
known_face_encodings = [
    zwh_face_encoding,
    chenduling_face_encoding,
    liujunbo_face_encoding
]

# 人物名称的集合
known_face_names = [
    "张文豪",
    "陈都灵",
    "刘军波"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while(cam.isOpened()):
    # 读取摄像头画面
    ret, frame = cam.read()
    if not ret:
        #等同于 if ret is not none
        break

    # 改变摄像头图像的大小，图像小，所做的计算就少
    small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)

    # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 默认为unknown
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.48)
            #阈值太低容易造成无法成功识别人脸，太高容易造成人脸识别混淆 默认阈值tolerance为0.6
            #print(matches)
            name = "Unknown"

            # if match[0]:
            #     name = "michong"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # 将捕捉到的人脸显示出来
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #由于我们检测到的帧被缩放到1/4大小，所以要缩小面位置
        top *= 3
        right *= 3
        bottom *= 3
        left *= 3

        # 矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
        ft = ft2.put_chinese_text('msyh.ttf')
        #引入ft2中的字体
        #加上标签
        #cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
       # cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)这是不输入汉字时可以用的代码

        frame = ft.draw_text(frame, (left+10 , bottom ), name, 20, (255, 255, 255))

         #cv2.imshow("frame",image)会出来两个框一个monitor 一个frame后者显示image但只有frame为true时才会显示
        #def draw_text(self, image, pos, text, text_size, text_color)

    cv2.imshow('monitor', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
