import cv2

v = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('frontal_face.xml')
frame = cv2.namedWindow('Distance Calculator', cv2.WINDOW_NORMAL)

font = cv2.FONT_HERSHEY_SIMPLEX

def find_length (y, h):
    return ( ((y + h) - y) ** 2) ** 0.5

def Calc_Dist (l):
    if l < 30 :
        return '130+'
    
    elif l >= 30 and l < 40:
        return 130 - (40 - l)
    
    elif l >= 40 and l < 50:
        return 120 - (50 - l)

    elif l >= 50 and l < 60:
        return 110 - (60 - l)
    
    elif l >= 60 and l < 70:
        return 100 - (70 - l)

    elif l >= 70 and l < 80:
        return 90 - (80 - l)

    elif l >= 80 and l < 90:
        return 80 - (90 - l)
    
    elif l >= 90 and l < 100:
        return 70 - (100 - l)
    
    elif l >= 100 and l < 120:
        return 60 - (120 - l)

    else :
        return '50-'

rec_color = (0, 0, 255) # RED COLOR
text_color = (255, 0, 0)  # BLUE COLOR

while True:
    ret, img = v.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale (gray)

    for x,y,w,h in faces :

        start = (x, y)
        end = (x + w, y + h)
        cv2.rectangle (img, start, end, rec_color, 1)

        length = find_length (y, h)
        distance = Calc_Dist (length)
        dist = str(distance) + ' cms'
            
        pos = (x - 20, y - 50)
        cv2.putText (img, dist, pos, font, 1, text_color, 2)

    cv2.imshow('Distance Calculator', img)

    if cv2.waitKey(10) is 27:
        break

v.release()
cv2.destroyAllWindows()



