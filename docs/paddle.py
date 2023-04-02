from paddlespeech.cli.asr.infer import ASRExecutor
class Speech():
    def __init__(self) -> None:
        self.ASRExecutor = ASRExecutor()
        self.device = 'gpu'
        pass

    def decode(self, audio_file):
        decode_recognition = self.ASRExecutor(audio_file=audio_file, device=self.device)
        return decode_recognition
    
speech = Speech()
txt =  speech.decode(myAudio.audio_path)
print(txt)