!pip install scipy==1.1.0
%matplotlib inline


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


import torchvision.transforms as transforms
import torchvision.models as models

import copy
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from scipy import misc
import tensorflow as tf
import tensorflow_hub as hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.
# Это всего лишь заготовка, поэтому не стесняйтесь менять имена функций, добавлять аргументы, свои классы и
# все такое.
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()   
    image = image.squeeze(0)      # функция для отрисовки изображения
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 
class StyleTransferModel:
    def __init__(self):
        # Сюда необходимо перенести всю иницализацию, вроде загрузки свеерточной сети и т.д.
        pass

    def transfer_style(self, content_img_stream, style_img_stream):
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
        # В первой итерации, когда вы переносите уже готовую модель из тетрадки с занятия сюда нужно просто
        # перенести функцию run_style_transfer (не забудьте вынести инициализацию, которая
        # проводится один раз в конструктор.

        # Сейчас этот метод просто возвращает не измененную content картинку
        # Для наглядности мы сначала переводим ее в тензор, а потом обратно
        #transform = transforms.Compose([
      #transforms.ToTensor(),
    #])
        #content=misc.toimage(self.process_image(content_img_stream)[0])
        #style=misc.toimage(self.process_image(content_img_stream)[0])
        #content.save(content_img_stream, format='PNG')
        #style.save(style_img_stream, format='PNG')
        #stylized_image = hub_module(transform(content[0]), transform(style[0]))[0]
        #return misc.toimage(stylized_image)
        plt.ion()   

        # отрисовка

        plt.figure()
        imshow(self.process_image(style_img_stream), title='Style Image')

        plt.figure()
        imshow(self.process_image(content_img_stream), title='Content Image')
        return misc.toimage(self.process_image(content_img_stream)[0])

    # В run_style_transfer используется много внешних функций, их можно добавить как функции класса
    # Если понятно, что функция является служебной и снаружи использоваться не должна, то перед именем функции
    # принято ставить _ (выглядит это так: def _foo() )
    # Эта функция тоже не является
    def process_image(self, img_stream):
        # TODO размер картинки, device и трансформации не меняются в течении всей работы модели,
        # поэтому их нужно перенести в конструктор!
        imsize = 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        loader = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат

        image = Image.open(img_stream)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)
        class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()#это константа. Убираем ее из дерева вычеслений
            self.loss = F.mse_loss(self.target, self.target )#to initialize with something

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input
class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target)# to initialize with something

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
cnn = models.vgg19(pretrained=True).features.to(device).eval()
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                #Переопределим relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        #выбрасываем все уровни после последенего styel loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses
def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        #добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        optimizer = optim.LBFGS([input_img.requires_grad_()]) 
        return optimizer
def run_style_transfer(cnn, normalization_mean, normalization_std,
                        content_img, style_img, input_img, num_steps=500,
                        style_weight=100000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values 
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                #взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img
class StyleTransferModel:
    def __init__(self):
        # Сюда необходимо перенести всю иницализацию, вроде загрузки свеерточной сети и т.д.
        pass

    def transfer_style(self, content_img_stream, style_img_stream):
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
        # В первой итерации, когда вы переносите уже готовую модель из тетрадки с занятия сюда нужно просто
        # перенести функцию run_style_transfer (не забудьте вынести инициализацию, которая
        # проводится один раз в конструктор.

        # Сейчас этот метод просто возвращает не измененную content картинку
        # Для наглядности мы сначала переводим ее в тензор, а потом обратно
        #transform = transforms.Compose([
      #transforms.ToTensor(),
    #])
        #content=misc.toimage(self.process_image(content_img_stream)[0])
        #style=misc.toimage(self.process_image(content_img_stream)[0])
        #content.save(content_img_stream, format='PNG')
        #style.save(style_img_stream, format='PNG')
        #stylized_image = hub_module(transform(content[0]), transform(style[0]))[0]
        #return misc.toimage(stylized_image)
        plt.ion()   

        # отрисовка

        plt.figure()
        imshow(self.process_image(style_img_stream), title='Style Image')

        plt.figure()
        imshow(self.process_image(content_img_stream), title='Content Image')
        input_img = self.process_image(content_img_stream).clone()
        return run_style_transfer(cnn, normalization_mean, normalization_std,
                        self.process_image(content_img_stream), self.process_image(style_img_stream), input_img, num_steps=500,
                        style_weight=100000, content_weight=1)
        #return misc.toimage(self.process_image(content_img_stream)[0])

    # В run_style_transfer используется много внешних функций, их можно добавить как функции класса
    # Если понятно, что функция является служебной и снаружи использоваться не должна, то перед именем функции
    # принято ставить _ (выглядит это так: def _foo() )
    # Эта функция тоже не является
    def process_image(self, img_stream):
        # TODO размер картинки, device и трансформации не меняются в течении всей работы модели,
        # поэтому их нужно перенести в конструктор!
        imsize = 128
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        loader = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат

        image = Image.open(img_stream)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)
!pip install python-telegram-bot
!pip uninstall tornado
!pip install tornado==4.5.3
#from model import StyleTransferModel
#from telegram_token import token
token = "1088329434:AAGMxB0cUxw24noU7S1GLkrC6qrbQi2Kseg"
from io import BytesIO
PHOTO, MSG = range(2)
# В бейзлайне пример того, как мы можем обрабатывать две картинки, пришедшие от пользователя.
import telegram
model = StyleTransferModel()
first_image_file = {}
def text(bot, update):
  update.message.reply_text('text', reply_markup=ReplyKeyboardRemove())
  return PHOTO
def skip_text():
  pass
def skip_photo():
  pass
def start(update, context):
  print("start")
  update.message.reply_text(
        'Hello, send me two pictures. The first is for the content and the second is for the style', reply_markup=ReplyKeyboardRemove())
  #bot.send_message(chat_id=update.message.chat_id, text='Help!')
  return PHOTO
def cancel():
  pass
def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)
def send_prediction_on_photo(update, context):
  bot = telegram.Bot(token=token)
  user = update.message.from_user
  photo_file = update.message.photo[-1].get_file()
  photo_file.download('user_photo.jpg')
  #logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    # Нам нужно получить две картинки, чтобы произвести перенос стиля, но каждая картинка приходит в
    # отдельном апдейте, поэтому в простейшем случае мы будем сохранять id первой картинки в память,
    # чтобы, когда уже придет вторая, мы могли загрузить в память уже сами картинки и обработать их.
    # Точно место для улучшения, я бы
  chat_id = update.message.chat_id
  print("Got image from {}".format(chat_id))

  # получаем информацию о картинке
  image_info = update.message.photo[-1]
  image_file = bot.get_file(image_info)

  if chat_id in first_image_file:
    # первая картинка, которая к нам пришла станет content image, а вторая style image
    content_image_stream = BytesIO()
    first_image_file[chat_id].download(out=content_image_stream)
    del first_image_file[chat_id]

    style_image_stream = BytesIO()
    image_file.download(out=style_image_stream)
    output = model.transfer_style(content_image_stream, style_image_stream)

    # теперь отправим назад фото
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to user")
  else:
    first_image_file[chat_id] = image_file
    return PHOTO
  return MSG

if __name__ == '__main__':
  from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
  from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          ConversationHandler)
  import logging

  logging.basicConfig(
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      level=logging.INFO)
  logger = logging.getLogger(__name__)
  updater = Updater(token=token,use_context=True, request_kwargs={'proxy_url': 'socks5h://163.172.152.192:1080'})
  #updater = Updater(token=token, use_context=True, request_kwargs={})
  conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            PHOTO: [MessageHandler(Filters.photo, send_prediction_on_photo),
                    CommandHandler('skip', skip_photo)],
            MSG: [MessageHandler(Filters.text, text),
                       CommandHandler('skip', skip_text)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
  dp = updater.dispatcher
  dp.add_handler(conv_handler)
  # log all errors
  dp.add_error_handler(error)

  # Start the Bot
  updater.start_polling()
  #pdater.dispatcher.add_handler(MessageHandler(Filters.photo, send_prediction_on_photo))
  updater.idle()
