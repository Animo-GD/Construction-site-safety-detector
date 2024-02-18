from ultralytics import YOLO
import cv2
import cvzone

nc = ["Gloves","Helmet","Human","Safety Boot","Safety Vest",
      "boots","glasses","gloves","Helmet","helmet","no boot","no boots","no gloves","no helmet","no vest","vest"]

vid = cv2.VideoCapture("Videos/worker3.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video5.mp4",fourcc,25,(3840,2160))
model = YOLO("best.pt")

while True:
    ret,frame = vid.read()
    if not ret:
        break

    results = model(frame,stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            w,h = x2-x1,y2-y1
            conf = int(box.conf[0]*100)/100
            cls = int(box.cls[0])
            class_name = nc[cls]
            myColor = (0,0,255)

            cvzone.putTextRect(frame,f"{class_name} {conf}",(max(0,x1),max(50,y1-20)))
            if "no" in class_name:
                myColor = (0,0,255)

            else:
                myColor = (0, 255, 0)
            cvzone.cornerRect(frame, [x1, y1, w, h], l=10, rt=5, colorR=myColor,colorC=myColor)

    cv2.imshow("video",frame)
    out.write(frame)
    if cv2.waitKey(1) == 27:
        break

out.release()
vid.release()
cv2.destroyAllWindows()