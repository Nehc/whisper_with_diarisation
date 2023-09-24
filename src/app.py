import sys
from io import StringIO
from diar import diarisation, wh_model
from googletrans import Translator
from pytube import YouTube
import gradio as gr

translator = Translator()

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def convert_aux(aud, lang, tr_lang):
  lang = None if lang == "Auto" else lang
  tr_lang = None if tr_lang == "Origin" else tr_lang
  with Capturing() as output:
    if lang and tr_lang:
      print(f'Setting language: {lang}, translate to: {tr_lang}')
      result = translator.translate(
                  wh_model.transcribe(aud, language=lang, 
                                   task='translate')["text"],
                  dest=tr_lang).text
    elif lang:
      print(f'Setting language: {lang}')
      result = wh_model.transcribe(aud, language=lang)["text"]
    elif tr_lang:
      print(f'Translate to: {tr_lang}')
      result = translator.translate(
                  wh_model.transcribe(aud, task='translate')["text"],
                  dest=tr_lang).text
    else:
      result = wh_model.transcribe(aud)["text"]
  try:
    output = f'<i>{output[0]}</i>\n' 
  except:
    output = ''
  return output+result


def get_yotube_aux(url):
  yt = YouTube(url)
  strms = yt.streams.filter(only_audio=True,mime_type="audio/mp4")
  strms[-1].download(filename='audio.mp3')


with gr.Blocks() as demo:
  
  with gr.Row():
    youturl = gr.Textbox(label="You-tube URL")
    sp_num = gr.Number(label="Spekers count", value=2)
  gr_audio = gr.Audio(label='Sound', type="filepath", interactive=True)
  with gr.Row():
    lang = gr.Dropdown(["Auto", "English", "Russian"], label="Original language")
    tr_lang = gr.Dropdown(["Origin", "English", "Russian"], label="Translated into")
  result = gr.Textbox(label="Output")
  with gr.Row():
    get_aux_btn = gr.Button("Get audio")
    text_btn = gr.Button("Get text")
    diar_btn = gr.Button("Diarisation")

  get_aux_btn.click(get_yotube_aux, inputs=[youturl], outputs=[gr_audio], api_name='get_aux')
  text_btn.click(convert_aux, inputs=[gr_audio,lang,tr_lang], outputs=[result], api_name='get_text')
  diar_btn.click(diarisation, inputs=[gr_audio,sp_num], outputs=[result], api_name='get_diar')

#demo.launch(server_name='0.0.0.0',server_port=7860)
demo.launch()
