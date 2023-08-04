import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from base64 import b64encode
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
warnings.filterwarnings("ignore")
from flask import Flask, request, render_template,send_file
from flask_ngrok import run_with_ngrok

template_folder = os.path.join(os.getcwd(), 'templates')
app = Flask(__name__, template_folder=template_folder)
app.secret_key = '17926381236'
run_with_ngrok(app)


@app.route('/')
def index():
    return render_template('index.html')

global model
def process_data():
  def transferAudio(sourceVideo, targetVideo):
      import shutil
      import moviepy.editor
      tempAudioFileName = "./temp/audio.mkv"

      # split audio from original video file and store in "temp" directory
      if True:

          # clear old "temp" directory if it exits
          if os.path.isdir("temp"):
              # remove temp directory
              shutil.rmtree("temp")
          # create new "temp" directory
          os.makedirs("temp")
          # extract audio from video
          os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

      targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
      os.rename(targetVideo, targetNoAudio)
      # combine audio file and new video file
      os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

      if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
          tempAudioFileName = "./temp/audio.m4a"
          os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
          os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
          if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
              os.rename(targetNoAudio, targetVideo)
              print("Audio transfer failed. Interpolated video will have no audio")
          else:
              print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

              # remove audio-less video
              os.remove(targetNoAudio)
      else:
          os.remove(targetNoAudio)

      # remove temp directory
      shutil.rmtree("temp")

  video=None
  exp=2
  output=None
  montage=False
  modelDir="train_log"
  fp16=False
  UHD=False
  scale=1.0
  skip=False
  fps=None
  png=False
  ext="mp4"
  img="input/"
  assert (not video is None or not img is None)
  if skip:
      print("skip flag is abandoned, please refer to issue #207.")
  if UHD and scale==1.0:
      scale = 0.5
  assert scale in [0.25, 0.5, 1.0, 2.0, 4.0]
  if not img is None:
      png = True

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.set_grad_enabled(False)
  if torch.cuda.is_available():
      torch.backends.cudnn.enabled = True
      torch.backends.cudnn.benchmark = True
      if(fp16):
          torch.set_default_tensor_type(torch.cuda.HalfTensor)

  try:
      try:
          try:
              from model.RIFE_HDv2 import Model
              model = Model()
              model.load_model(modelDir, -1)
              print("Loaded v2.x HD model.")
          except:
              from train_log.RIFE_HDv3 import Model
              model = Model()
              model.load_model(modelDir, -1)
              print("Loaded v3.x HD model.")
      except:
          from model.RIFE_HD import Model
          model = Model()
          model.load_model(modelDir, -1)
          print("Loaded v1.x HD model")
  except:
      from model.RIFE import Model
      model = Model()
      model.load_model(modelDir, -1)
      print("Loaded ArXiv-RIFE model")
  model.eval()
  model.device()

  if not video is None:
      videoCapture = cv2.VideoCapture(video)
      fps = videoCapture.get(cv2.CAP_PROP_FPS)
      tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
      videoCapture.release()
      if fps is None:
          fpsNotAssigned = True
          fps = fps * (2 ** exp)
      else:
          fpsNotAssigned = False
      videogen = skvideo.io.vreader(video)
      lastframe = next(videogen)
      fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
      video_path_wo_ext, ext = os.path.splitext(video)
      print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, ext, tot_frame, fps, fps))
      if png == False and fpsNotAssigned == True:
          print("The audio will be merged after interpolation process")
      else:
          print("Will not merge audio because using png or fps flag!")
  else:
      videogen = []
      for f in os.listdir(img):
          if 'png' in f:
              videogen.append(f)
      tot_frame = len(videogen)
      videogen.sort(key= lambda x:int(x[:-4]))
      lastframe = cv2.imread(os.path.join(img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
      videogen = videogen[1:]
  h, w, _ = lastframe.shape
  vid_out_name = None
  vid_out = None
  if png:
      if not os.path.exists('vid_out'):
          os.mkdir('vid_out')
  else:
      if output is not None:
          vid_out_name = output
      else:
          vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** exp), int(np.round(fps)), ext)
      vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps, (w, h))

  def clear_write_buffer(write_buffer):
      cnt = 0
      while True:
          item = write_buffer.get()
          if item is None:
              break
          if png:
              cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
              cnt += 1
          else:
              vid_out.write(item[:, :, ::-1])

  def build_read_buffer(read_buffer, videogen):
      try:
          for frame in videogen:
              if not img is None:
                    frame = cv2.imread(os.path.join(img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
              if montage:
                    frame = frame[:, left: left + w]
              read_buffer.put(frame)
      except:
          pass
      read_buffer.put(None)

  def make_inference(I0, I1, n):

      middle = model.inference(I0, I1, scale)
      if n == 1:
          return [middle]
      first_half = make_inference(I0, middle, n=n//2)
      second_half = make_inference(middle, I1, n=n//2)
      if n%2:
          return [*first_half, middle, *second_half]
      else:
          return [*first_half, *second_half]

  def pad_image(img):
      if(fp16):
          return F.pad(img, padding).half()
      else:
          return F.pad(img, padding)

  if montage:
      left = w // 4
      w = w // 2
  tmp = max(32, int(32 / scale))
  ph = ((h - 1) // tmp + 1) * tmp
  pw = ((w - 1) // tmp + 1) * tmp
  padding = (0, pw - w, 0, ph - h)
  pbar = tqdm(total=tot_frame)
  if montage:
      lastframe = lastframe[:, left: left + w]
  write_buffer = Queue(maxsize=500)
  read_buffer = Queue(maxsize=500)
  _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
  _thread.start_new_thread(clear_write_buffer, (write_buffer,))

  I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
  I1 = pad_image(I1)
  temp = None # save lastframe when processing static frame

  while True:
      if temp is not None:
          frame = temp
          temp = None
      else:
          frame = read_buffer.get()
      if frame is None:
          break
      I0 = I1
      I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
      I1 = pad_image(I1)
      I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
      I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
      ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

      break_flag = False
      if ssim > 0.996:        
          frame = read_buffer.get() # read a new frame
          if frame is None:
              break_flag = True
              frame = lastframe
          else:
              temp = frame
          I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
          I1 = pad_image(I1)
          I1 = model.inference(I0, I1, scale)
          I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
          ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
          frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
      
      if ssim < 0.2:
          output = []
          for i in range((2 ** exp) - 1):
              output.append(I0)
          '''
          output = []
          step = 1 / (2 ** exp)
          alpha = 0
          for i in range((2 ** exp) - 1):
              alpha += step
              beta = 1-alpha
              output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
          '''
      else:
          output = make_inference(I0, I1, 2**exp-1) if exp else []

      if montage:
          write_buffer.put(np.concatenate((lastframe, lastframe), 1))
          for mid in output:
              mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
              write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
      else:
          write_buffer.put(lastframe)
          for mid in output:
              mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
              write_buffer.put(mid[:h, :w])
      pbar.update(1)
      lastframe = frame
      if break_flag:
          break

  if montage:
      write_buffer.put(np.concatenate((lastframe, lastframe), 1))
  else:
      write_buffer.put(lastframe)
  import time
  while(not write_buffer.empty()):
      time.sleep(0.1)
  pbar.close()
  if not vid_out is None:
      vid_out.release()

  # move audio to new video file if appropriate
  if png == False and fpsNotAssigned == True and not video is None:
      try:
          transferAudio(video, vid_out_name)
      except:
          print("Audio transfer failed. Interpolated video will have no audio")
          targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
          os.rename(targetNoAudio, vid_out_name)

  return "Data processed successfully!"

def save(input_folder):
    command = f"ffmpeg -r 1 -i {input_folder}/%07d.png -vcodec libx264 -y /content/ECCV2022-RIFE/temp/output_video.mp4"
    os.system(command)
    return '/content/ECCV2022-RIFE/temp/output_video.mp4'

# Example usage
input_folder = "/content/ECCV2022-RIFE/vid_out/"
save(input_folder)

import time
upload_path = "/content/ECCV2022-RIFE/input/"
@app.route('/upload_folder', methods=['GET', 'POST'])
def process():
      if 'folder' not in request.files:
        return "No folder part"

      folder = request.files.getlist('folder')
    
      for file in folder:
          if file.filename == '':
              continue
          
          filename = os.path.basename(file.filename)
          filepath = os.path.join(upload_path, filename)
          file.save(filepath)
      if request.method == 'POST':
          result = process_data()
      input_folder = "/content/ECCV2022-RIFE/vid_out/"
      output_path=save(input_folder)

      video_path= get_data_url(output_path)

  
      return render_template('index.html',video_path=video_path)


def get_data_url(video_path):
    video_bytes = open(video_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(video_bytes).decode()
    return data_url

if __name__ == '__main__':
    app.run()
