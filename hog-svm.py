import cv2
import dlib
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
from imutils.video import  WebcamVideoStream
import numpy as np
def ear(eye):    # eye aspect ratio
    vertical1_dist = dist.euclidean(eye[1], eye[5])
    vertical2_dist = dist.euclidean(eye[2], eye[4])
    horizontal_dist = dist.euclidean(eye[0], eye[3])
    aspect_ratio = (vertical1_dist + vertical2_dist) / (2.0 * horizontal_dist)
    return aspect_ratio
threshold = 0.30
framenos = 44
age_list = ['(0,2)', '(4,6)', '(8,12)', '(15,18)', '(21,32)', '(38,43)', '(48,53)', '(60,100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    return age_net
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
(lx,ly) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rx,ry) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
file =  WebcamVideoStream(src=0).start()
first_frame = file.read()
(iheight,iwidth) = first_frame.shape[:2]
print(iheight,iwidth)
def det_to_bb(det):
    x = det.left()
    y = det.top()
    w = det.right() 
    h = det.bottom() 
    return (x, y, w, h)
def video_detector(age_net):
    counter = 0
    while True:
        frame = file.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect = detector(gray)
        if (len(detect)>0) :
            for (i,det) in enumerate(detect):
                ishape = predictor(gray,det)
                ishape = face_utils.shape_to_np(ishape)
                (x, y, w, h) = face_utils.rect_to_bb(det)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                key = cv2.waitKey(1)
                #for (x,y) in ishape:
                    #cv2.circle(frame,(x,y),1,(0,255,0),1)
                
                face_img = frame[y:y + h, h:h + w].copy()
                blob=cv2.dnn.blobFromImage(face_img,1,(244,244),MODEL_MEAN_VALUES,swapRB=True)
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                overlay_text = "%s" % (age)
                cv2.putText(frame, overlay_text,(x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
              
                leftEye = ishape[lx:ly]
                rightEye = ishape[rx:ry]

                leftEar = ear(leftEye)
                rightEar = ear(rightEye)
                avgEar = (leftEar+rightEar)/2
                if avgEar < threshold:
                    counter += 1
                    if counter >= framenos:
                        cv2.putText(frame, "DROWSY",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        cv2.putText(frame, "NOT ELIGIBLE FOR DRIVING-Drowsy",(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    
                            
                            
                else:
                    counter = 0
                    cv2.putText(frame, "NOT DR0WSY",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if overlay_text in (age_list[0],age_list[1],age_list[2],age_list[3]):
                        cv2.putText(frame, "NOT ELIGIBLE FOR DRIVING-Underage and Drowsy",(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    else:
                        cv2.putText(frame, "ELIGIBLE FOR DRIVING",(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        else:
            cv2.putText(frame, "NO FACE PRESENT",(20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('original frame',frame)
        frame = cv2.resize(frame,(iwidth,iheight))
        key = cv2.waitKey(1)
        if key == ord('q'):
            file.stop()
            break
    file.stream.release()
    cv2.destroyAllWindows()
def main():
    age_net = load_caffe_models()
    video_detector(age_net)  

if __name__ == "__main__":
    main()