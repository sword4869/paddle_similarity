# -*- coding: utf-8 -*-
import sys
from loguru import logger
import json


sys.path.append('dependencies')

from similarities.fastsim import AnnoySimilarity

logger.remove()
logger.add(sys.stderr, level="INFO")



class QAMatching:
    def __init__(self, corpus_path='data/corpora/qa_teachers.json') -> None:
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
        with open(path, encoding='utf-8') as f:
            qa = json.loads(f.read())
        
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


def main():
    engine = QAMatching()

    while True:
    # if True:
        # sentence = '档案价值是什么'
        sentence = input('Q: ')
        if engine.is_qa_matching(sentence):
            print(engine.best_question)
            print(engine.ans_txt)
            print(engine.ans_wav)
        else:
            print('failure...')

if __name__ == '__main__':
    main()