import logging
from telegram.ext import Updater , CommandHandler , MessageHandler , Filters
from fastai.vision.all import load_learner
import matplotlib.pyplot as plt
import cv2
import pathlib 
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__) 

def start(update, context):
    update.message.reply_text(
        "Bot by @IshantKukreti  \n\n "
        "Just send me a photo of you and I will tell you's mood 😏 \n"
        
    )


def help_command(update, context):
    update.message.reply_text('My only purpose is to tell your mood . Send a photo')

# load Model 

def load_model():
    global model 
    model = load_learner('data/emo7.pkl')
    print('model loaded')

def detect_emo(update , context):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    img_raw = plt.imread('user_photo.jpg')
    gray = cv2.cvtColor(img_raw , cv2.COLOR_BGR2GRAY)
    pred = model.predict(gray)[0]
    # print(pred)
    update.message.reply_text("you look " )
    update.message.reply_text(str(pred))

def main():
    load_model()
    updater = Updater(token = "1952840230:AAGggG_beLMrs9Re5GIOVChjoskV5tldHMs", use_context= True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help_command))
    dp.add_handler(MessageHandler(Filters.photo, detect_emo))
    updater.start_polling()
    updater.idle()


if __name__=='__main__':
    main()
