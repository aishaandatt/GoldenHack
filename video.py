# import opencv
import cv2
# import numpy
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
# import Client from twilio API
from twilio.rest import Client

model = load_model(
    '/Users/aishaandatt/Downloads/goldenhack/bird.h5')
# To read webcam
video = cv2.VideoCapture(0)
# Type of classes or names of the labels that we considered
train_dir = "/Users/aishaandatt/Downloads/archive/train"
generator = ImageDataGenerator()
train_ds = generator.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32)
classes = list(train_ds.class_indices.keys())
# To execute the program repeatedly using while loop
while(1):
    success, frame = video.read()
    cv2.imwrite("image.jpg", frame)
    img = image.load_img("image.jpg", target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict_classes(x)
    p = pred[0]
    # print(pred)
    cv2.putText(frame, "predicted  class = "+str(classes[p]), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    pred = model.predict_classes(x)
    if pred[0] == 2:
        # twilio account ssid
        account_sid = 'AC4c30xxxxxxx'
        # twilo account authentication toke
        auth_token = 'd22xxxxxx'
        client = Client(account_sid, auth_token)

        message = client.messages \
            .create(
                body='Danger!. Wild animal is detected, stay alert',
                from_=' +12293515154',  # the free number of twilio
                to='+919149492527')
        print(message.sid)
        print('Danger!!')
        print('Animal Detected')
        print('SMS sent!')
        # break
    else:
        print("No Danger")
       # break
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

video.release()
cv2.destroyAllWindows()
