#Banking Application

import cv2
import os
import numpy as np
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle


def getAccandFaceData():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # For each person, enter one numeric face id
    with open('bankaccdb','rb') as file:
        AccountInfo=pickle.load(file)
        start=pickle.load(file)
        name=input("Enter Account Holder Name:\n")
        AccountInfo[start]=[name,500];
    with open('bankaccdb','wb')as file:
        pickle.dump(AccountInfo,file)
        pickle.dump(start+1,file)
        print("Your Account Number:",start)
        print("Thanks for Account creating with us...")
    print("\n[INFO] Initializing face capture. Look the camera and wait ...")
	# Initialize individual sampling face count
    count = 0

    while(True):

        ret, img = cam.read()
        img = cv2.flip(img, 1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1

        # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(start) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            break

    cam.release()
    cv2.destroyAllWindows()




def trainFaceData():
# Path for face image database
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples, ids

    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi



def processFaceData():
    with open('bankaccdb', 'rb')as file:
        AccountInfo = pickle.load(file)
        start=pickle.load(file)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
  #  id = 0
    choice = int(input("1.Withdrawal\n2.Transfer Amount\n3.Balance Enquiry\n4.Close\n"))
    if choice == 1:
        accno = int(input("Enter your account number for the withdrawal\n"))
        if accno not in AccountInfo.keys():
            print("Account Not Found. Create a new one with option 1 in main menu")
            return
        withdrawAmt = int(input("Enter the amount to withdraw\n"))
        if AccountInfo[accno][1] < withdrawAmt:
            print("Insufficient Fund")
            return
    elif choice == 2:
        accno = int(input("Enter your account number:\n"))
        receiveraccno = int(input("Enter the account number to transfer the amount\n"))
        if accno not in AccountInfo.keys() or receiveraccno not in AccountInfo.keys():
            print("Account Not Found. Create a new one with option 1 from main menu")
            return
        transferamt=int(input("Enter the amount to transfer\n"))
        if transferamt > AccountInfo[accno][1]:
            print("Insufficient Fund\n")
            return
    elif choice == 3:
        accno = int(input("Enter your Account Number for Balance Enquiry:\n"))
        if accno not in AccountInfo.keys():
            print("Account not Found. Create a new one with option 1 from main menu")
            return
    elif choice == 4:
        print("Returning to main menu")
        return
    else:
        print("Enter valid option returning to main menu...\n")
        return


# Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, img = cam.read()
        img = cv2.flip(img, 1) # Flip vertically

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)))
        withdrawan = 0
        for(x, y, w, h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100 and id == accno:
               # confidence = "  {0}%".format(round(100 - confidence))
                if choice == 1:
                    AccountInfo[id][1] -= withdrawAmt
                elif choice == 2:
                    AccountInfo[id][1] -= transferamt
                withdrawan = 1
                break
            else:
                break
        cv2.imshow('camera', img)

        if withdrawan == 1 and choice == 2 and id == accno:
            AccountInfo[receiveraccno][1] += transferamt
            print("Amount transferred")
            break
        if withdrawan == 1 and choice == 1 and id == accno:
            print("Amount has been withdrawn\n")
            break
        if withdrawan == 1 and choice == 3 and id == accno:
            print("Account Holder name:",AccountInfo[id][0])
            print("Your Account Balance:", AccountInfo[id][1])
            break
       # if id != accno:
        #    print("Face didn't matched try again...",id,accno)
         #   break
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    with open('bankaccdb', 'wb')as file:
        pickle.dump(AccountInfo, file)
        pickle.dump(start, file)
    

while 1:
    print("--------------------Banking Service--------------------\n")
    optionToProceed = int(input("1. Account creation and Face Enroll for your Account Security\n2. Deposit\n3. Withdrawal, Amount Transfer, Balance Enquiry\n4. Exit\n"))
    if optionToProceed == 4:
        break
    elif optionToProceed == 1:
        getAccandFaceData()
        trainFaceData()
    elif optionToProceed == 2:
        accno = int(input("Enter the account number to deposit amount:\n"))
        with open('bankaccdb', 'rb') as file:
            AccountInfo = pickle.load(file)
            start=pickle.load(file)
            if accno not in AccountInfo.keys():
                print('Account Not Found\n')
            else:
                value = int(input("Enter the amount to be deposited\n"))
                AccountInfo[accno][1] += value
                print("Amount deposited")
        with open('bankaccdb', 'wb')as file:
            pickle.dump(AccountInfo, file)
            pickle.dump(start,file)
    elif optionToProceed == 3:
        processFaceData()
    else:
        print("Try Again... Please Insert Valid Option\n")