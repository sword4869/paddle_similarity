import time 
from paddlespeech.cli.asr.infer import ASRExecutor
import json

import sys
import os
import wave
import pyaudio

import sys
from loguru import logger
import json


sys.path.append('dependencies/similarities')

from similarities.fastsim import AnnoySimilarity

logger.remove()
logger.add(sys.stderr, level="INFO")

class MyAudio:
    # 定义音频属性
    frames_per_buffer = 1024
    pyaudio_format = pyaudio.paInt16
    n_channels = 1
    sample_rate = 16000
    audio_path = "output.wav"  
    
    def __init__(self):
        pass
        
    def record_start(self):
        ### 
        # 不定时长录音之启动
        ###

        # 创建PyAudio对象
        self.p = pyaudio.PyAudio()

        # 创建wave对象        
        self.wf = wave.open(self.audio_path, 'wb')
        self.wf.setnchannels(self.n_channels)
        self.wf.setsampwidth(self.p.get_sample_size(self.pyaudio_format))
        self.wf.setframerate(self.sample_rate)


        def callback(in_data, frame_count, time_info, status):
            self.wf.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        # 打开数据流
        self.stream = self.p.open(format=self.pyaudio_format,
                        channels=self.n_channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.frames_per_buffer,
                        stream_callback=callback)

        print("* recording")

        # 开始录音
        # start_stream()之后stream会开始调用callback函数
        self.stream.start_stream()


    def record_stop(self):   
        ### 
        # 不定时长录音之停止
        ###
        #      
        self.stream.stop_stream()   # 停止数据流
        self.stream.close()
        self.p.terminate()          # 关闭 PyAudio
        self.wf.close()             # 关闭 wave

        print("* done recording")


    def record_play(self):
        ### 
        # 播放录音
        ###

        wf=wave.open(self.audio_path, 'rb')

        p=pyaudio.PyAudio()
        stream=p.open(
            format=p.get_format_from_width(width=wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        print('* play...')
        stream.write(wf.readframes(wf.getnframes()))
        
        stream.stop_stream()   # 停止数据流
        stream.close()
        p.terminate()          # 关闭 PyAudio
        wf.close()             # 关闭 wave

        print('* done play')



class Speech():
    def __init__(self) -> None:
        self.ASRExecutor = ASRExecutor()
        self.device = 'gpu'
        pass

    def decode(self, audio_file):
        decode_recognition = self.ASRExecutor(audio_file=audio_file, device=self.device)
        return decode_recognition
    

class QAMatching:
    def __init__(self, corpus_path='data/qa_3.json') -> None:
        for i in range(2):
            try:
                # self.model=AnnoySimilarity(model_name_or_path="shibing624/text2vec-base-chinese")
                self.model=AnnoySimilarity(model_name_or_path="data/models--shibing624--text2vec-base-chinese/snapshots/13ee917482d459b4d82986e324d2adf8ce6692df/")
                break
            except Exception as e:
                if i < 1:
                    pass
                else:
                    raise e
                
        self.basic_threshold = 0.6
        self.topn = 3

        self.load_corpus(corpus_path)

    def load_corpus(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            qa = json.load(f)
        
        questions = []
        self.questions_accumulative_ids = []
        self.ans_txts = []
        self.ans_wavs = []
        
        questions_accumulative_id = 0
        for item in qa['keyword']['*']:
            questions += item['questions']
            questions_accumulative_id = questions_accumulative_id + len(item['questions'])
            self.questions_accumulative_ids.append(questions_accumulative_id)
            self.ans_txts.append(item['ans_txt'])
            self.ans_wavs.append(item['ans_wav'])
            
        m_corpus = {i:v for i, v in enumerate(questions)}
        
        self.model.add_corpus(m_corpus)
        # print(self.model.corpus)

        self.model.build_index()


    def is_qa_matching(self, sentence):
        self.best_question = None
        self.ans_txt = None
        self.ans_wav = None

        res = self.model.most_similar(queries=sentence, topn=self.topn)

        
        best_questions_id, best_score = None, 0
        # 每个sentence
        for q_id, c in res.items():
            # 每个匹配项
            for questions_id, score in c.items():
                question = self.model.corpus[questions_id]
                print(f'[{questions_id}]: question={question}, score={score:.2f}')
                
                if best_questions_id is None:
                    self.best_question = question
                    best_questions_id = questions_id
                    best_score = score

        if best_score < self.basic_threshold:
            return False

        self.best_ans_id = None
        # print(self.questions_accumulative_ids)
        for i, id in enumerate(self.questions_accumulative_ids):
            # print(i, id , best_questions_id)
            if id <= best_questions_id:
                continue
            else:
                self.best_ans_id = i
                break

        self.ans_txt = self.ans_txts[self.best_ans_id]
        self.ans_wav = self.ans_wavs[self.best_ans_id]
    
        return True



json_path = 'data/qa.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
data = data['keyword']['*']
print(len(data))


myAudio = MyAudio()
speech = Speech()
engine = QAMatching(json_path)
t = []

for index, qa in enumerate(data):
    print(f'* [{index}]', qa['questions'])
    time.sleep(2)
    # 不定时录音
    myAudio.record_start()
    stop = input('* waiting')
    print('* stop')
    myAudio.record_stop()

    # 播放录音
    # myAudio.record_play()

    sentence =  speech.decode(myAudio.audio_path)
    print(f'* q: {sentence}')

    if engine.is_qa_matching(sentence):
        print(engine.best_question)
    else:
        print('* !!!!!!!!!!!!!!!...')

    c = input('j or k')
    if c == 'j':
        t.append(True)
    elif c == 'k':
        t.append(False)
    else:
        print('******* wrong')

    print(t)
    print('---------------')