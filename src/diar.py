import subprocess, datetime, contextlib
import wave, torch, whisper
from sklearn.cluster import SpectralClustering
#from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import numpy as np
import pandas as pd

wh_model = whisper.load_model("small")

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-resnet-voxceleb",
    #"speechbrain/spkrec-ecapa-voxceleb",
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

audio_proc = Audio()

def time(secs):
  return datetime.timedelta(seconds=round(secs))


def convert_to_wav(path):
  new_path = '.'.join(path.split('.')[:-1]) + '_mono.wav'
  try:
    subprocess.call(['ffmpeg', '-i', path, '-ac', '1', new_path, '-y'])
  except Exception as e:
    return path, f'Error: Could not convert file to .wav ({e})'
  return new_path, 'file converted'


def get_duration(path):
  with contextlib.closing(wave.open(path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    return frames / float(rate)


def segment_embedding(path, segment, duration):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform,_ = audio_proc.crop(path, clip)
  return embedding_model(waveform[None])


def make_embeddings(path, segments, duration):
  embeddings = np.zeros(shape=(len(segments), 192))
  for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(path, segment, duration)
  return np.nan_to_num(embeddings)


def add_speaker_labels(segments, embeddings, num_speakers=2, speakers=None):
  if speakers:
      num_speakers = len(speakers)
  #clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
  clustering = SpectralClustering(num_speakers).fit(embeddings)
  labels = clustering.labels_
  for i in range(len(segments)):
    if speakers:
      segments[i]["speaker"] = speakers[labels[i]]
    else:
      segments[i]["speaker"] = 'SPEAKER-' + str(labels[i] + 1)


def group_df_by_speaker(df):
  df['change_speaker'] = df['speaker'].ne(df['speaker'].shift()).cumsum()
  agg_functions = {'start': 'min', 'end': 'max', 'text': ' '.join}
  result_df = df.groupby(['change_speaker', 'speaker'], as_index=False).agg(agg_functions)
  # Функция для преобразования секунд в формат (h.mm.ss)
  def format_seconds_to_hhmmss(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):01d}:{int(minutes):02d}:{int(seconds):02d}"
  # Примените функцию к столбцам start и end
  result_df['start'] = result_df['start'].apply(format_seconds_to_hhmmss)
  result_df['end'] = result_df['end'].apply(format_seconds_to_hhmmss)
  # Удалите временный столбец
  result_df.drop(columns=['change_speaker'], inplace=True)
  return result_df


def get_output(segments):
  output = ''
  for (i, segment) in enumerate(segments):
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
      if i != 0:
        output += '\n\n'
      output += segment["speaker"] + ' ' + str(time(segment["start"])) + '\n\n'
    output += segment["text"][1:] + ' '
  return output


def format_output(row):
    sp_lenth = 10
    return f"{row['speaker'].ljust(sp_lenth)}({row['start']}-{row['end']}): {row['text']}"


def diarisation(aud, sp_num=2):
  result = wh_model.transcribe(aud)
  segments = result["segments"]
  wav_file, res = convert_to_wav(aud); print(res) 
  emb = make_embeddings(wav_file, segments, get_duration(wav_file))
  add_speaker_labels(segments, emb, num_speakers=int(sp_num))
  df = pd.DataFrame.from_records(segments,index = 'id', 
                    columns=['id','speaker','text','start','end'])
  df.to_csv('.'.join(aud.split('.')[:-1]) + '.csv')
  df = group_df_by_speaker(df)
  formated_text = '\n'.join(df.apply(format_output, axis=1))
  return formated_text
