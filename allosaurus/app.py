from allosaurus.am.utils import *
from pathlib import Path
from allosaurus.audio import read_audio, read_samples
from allosaurus.pm.factory import read_pm
from allosaurus.am.factory import read_am
from allosaurus.lm.factory import read_lm
from allosaurus.bin.download_model import download_model
from allosaurus.model import resolve_model_name, get_all_models
from argparse import Namespace
from datetime import datetime
import webrtcvad

def read_recognizer(inference_config_or_name='latest', alt_model_path=None):
    if alt_model_path:
        if not alt_model_path.exists():
            download_model(inference_config_or_name, alt_model_path)
    # download specified model automatically if no model exists
    if len(get_all_models(alt_model_path)) == 0:
        download_model('latest', alt_model_path)

    # create default config if input is the model's name
    if isinstance(inference_config_or_name, str):
        model_name = resolve_model_name(inference_config_or_name, alt_model_path)
        inference_config = Namespace(model=model_name, device_id=-1, lang='ipa', approximate=False, prior=None)
    else:
        assert isinstance(inference_config_or_name, Namespace)
        inference_config = inference_config_or_name

    if alt_model_path:
        model_path = alt_model_path / inference_config.model
    else:
        model_path = Path(__file__).parent / 'pretrained' / inference_config.model

    if inference_config.model == 'latest' and not model_path.exists():
        download_model(inference_config, alt_model_path)

    assert model_path.exists(), f"{inference_config.model} is not a valid model"

    # create pm (pm stands for preprocess model: audio -> feature etc..)
    pm = read_pm(model_path, inference_config)

    # create am (acoustic model: feature -> logits )
    am = read_am(model_path, inference_config)

    # create lm (language model: logits -> phone)
    lm = read_lm(model_path, inference_config)

    return Recognizer(pm, am, lm, inference_config)

class Recognizer:

    def __init__(self, pm, am, lm, config):

        self.pm = pm
        self.am = am
        self.lm = lm
        self.config = config

    def is_available(self, lang_id):
        # check whether this lang id is available

        return self.lm.inventory.is_available(lang_id)

    def recognize(self, filename, lang_id='ipa', topk=1, emit=1.0, timestamp=False):
        # recognize a single file

        assert str(filename).endswith('.wav'), "only wave file is supported in allosaurus"

        # load wav audio
        audio = read_audio(filename)

        # extract feature
        feat = self.pm.compute(audio)

        # add batch dim
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)

        tensor_batch_feat, tensor_batch_feat_len = move_to_tensor([feats, feat_len], self.config.device_id)

        tensor_batch_lprobs = self.am(tensor_batch_feat, tensor_batch_feat_len)

        if self.config.device_id >= 0:
            batch_lprobs = tensor_batch_lprobs.cpu().detach().numpy()
        else:
            batch_lprobs = tensor_batch_lprobs.detach().numpy()

        token = self.lm.compute(batch_lprobs[0], lang_id, topk, emit=emit, timestamp=timestamp)
        return token

    def streaming_recognize(self, config):
        return RecognizeStream(self, config)

    def recognize_samples(self, samples, lang_id='ipa', topk=1, emit=1.0, timestamp=False):
        
        audio_config = Namespace(channels=1, frame_rate=8000, sample_width=8)

        # load wav audio
        audio = read_samples(samples, audio_config)

        # extract feature
        feat = self.pm.compute(audio)

        # add batch dim
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)

        tensor_batch_feat, tensor_batch_feat_len = move_to_tensor([feats, feat_len], self.config.device_id)

        tensor_batch_lprobs = self.am(tensor_batch_feat, tensor_batch_feat_len)

        if self.config.device_id >= 0:
            batch_lprobs = tensor_batch_lprobs.cpu().detach().numpy()
        else:
            batch_lprobs = tensor_batch_lprobs.detach().numpy()

        token = self.lm.compute(batch_lprobs[0], lang_id, topk, emit=emit, timestamp=timestamp)
        return token

class ObjectWithEvents(object):
    callbacks = None

    def on(self, event_name, callback):
        if self.callbacks is None:
            self.callbacks = {}

        if event_name not in self.callbacks:
            self.callbacks[event_name] = [callback]
        else:
            self.callbacks[event_name].append(callback)

    def trigger(self, event_name, *args, **kwargs):
        if self.callbacks is not None and event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                callback(*args, **kwargs)
                
class RecognizeStream(ObjectWithEvents):

    def __init__(self, recognizer, config):
        self.recognizer = recognizer
        self.config = config
        self.lang = config.lang
        print(self.lang)
        # self.topk = config.topk
        # self.emit = config.emit
        # self.timestamp = config.timestamp
        self.audio_config = Namespace(channels=1, frame_rate=8000, sample_width=8)
        self.vad = webrtcvad.Vad()

        self.current_samples = np.empty(0, int)
        self.last_phone_at = datetime.now()
        self.last_cache_offset = 0
        self.current_phones = ""
        self.current_transcript = ""
        self.phone_timings = {}
        self.silence_time = 0
        self.previous_phones = ""
        self.previous_phone_timings = []

        self.total_silence_ms = 0
        self.samples_processed = False
        self.total_speech_ms = 0
        self.samples_processed = False
        self.speech_found = False
        self.samples = np.empty(0, int)
        self.offset = 0
        self.frame_offset = 0

        self.in_speech = False

    def recognize_new_samples(self, samples):
        
        phone_and_timing_string = self.recognizer.recognize_samples(samples, lang_id=self.lang, timestamp=True)
        phone_and_timing_list = phone_and_timing_string.split('\n')
        #print('Recognized: ', phones)
        #if phones != self.current_phones:
        
        self.current_phone_timings = []
        last_end_time_ms = 0
        self.current_phones = ""

        # loop through phone_and_timing_list
        for phone_and_timing in phone_and_timing_list:
            if phone_and_timing == '':
                continue

            start_time_s_string, length_s_string, phone = phone_and_timing.split(' ')
            start_time_ms = round(float(start_time_s_string) * 1000)
            length_ms = round(float(length_s_string) * 1000)
            #print('phone: ', phone)
            #print('timing: ', timing)
            
            self.current_phone_timings.append({
                'phone': phone,
                'start_time_ms': start_time_ms,
                'length_ms': length_ms,
                'end_time_ms': start_time_ms + length_ms,
                'delay_from_last_ms': start_time_ms - last_end_time_ms
            })

            last_end_time_ms = start_time_ms + length_ms
            self.current_phones += phone + ' '
        
        self.last_phone_at = datetime.now()
        #    self.silence_time = 0
        self.current_phones = self.current_phones.strip()

        print("Recognized:", self.current_phones, self.current_phone_timings)

        new_transcript = self.previous_phones + " " + self.current_phones
        new_transcript = new_transcript.strip()

        # else:
        #     self.silence_time += (1.0 / self.audio_config.frame_rate) * len(samples)
        #     print(self.silence_time)
        #     if self.silence_time > 0.3 and len(self.current_phones) > 0:
        #         self.previous_phones += self.current_phones + " "
        #         back_count = int(self.audio_config.frame_rate * (50 / 1000.0))
        #         self.current_samples = self.current_samples[-back_count:]
        #         self.last_phone_at = datetime.now()
        #         self.current_phones = ""
        #         self.silence_time = 0

        print("New transcript:", new_transcript)

        self.trigger('data', new_transcript)
        return new_transcript

    def cache_phones_if_ready(self, end_offset):
        if not self.speech_found or self.current_phones == "":
            return

        min_duration = 3000
        min_samples = int(8000 * min_duration / 1000)

        if end_offset - self.offset < min_samples:
            print("Not ready to cache", end_offset - self.offset, min_samples)
            return

        min_end_time_ms = 2000
        min_delay_from_last_ms = 50

        # How far into the current samples will we cache?
        cache_offset = 0
        cache_char_index = -1

        # loop through current_phone_timings with index
        for i, phone_timing in enumerate(self.current_phone_timings):
            if phone_timing['end_time_ms'] < min_end_time_ms:
                continue

            if phone_timing['delay_from_last_ms'] < min_delay_from_last_ms:
                continue

            cache_offset = int(8000 * phone_timing['end_time_ms'] / 1000)
            cache_char_index = i
            break

        if cache_offset == 0:
            return

        print("ready to cache", cache_offset, cache_char_index, self.current_phone_timings[cache_char_index])
        print("offset before cache:", self.offset)
        self.cache_phones(self.offset + cache_offset, (cache_char_index+1)*2)
        print("offset after cache:", self.offset)

        return        

    def cache_phones_if_ready_bak(self, end_offset):
        if not self.speech_found or self.current_phones == "":
            return

        min_duration = 3000
        min_samples = int(8000 * min_duration / 1000)

        if end_offset - self.offset < min_samples:
            print("Not ready to cache", end_offset - self.offset, min_samples)
            return

        min_age_duration = 2000
        min_age_samples = int(8000 * min_age_duration / 1000)
        min_age_offset = end_offset - min_age_samples        
        
        cache_offset = 0

        print("Checking timings for cacheability", min_age_offset, self.offset)

        for phone_offset in self.phone_timings:
            if phone_offset < self.offset:
                # These phones are already cached
                continue
            if phone_offset > min_age_offset:
                # These phones are too new
                continue
            if phone_offset > cache_offset:
                cache_offset = phone_offset

        if cache_offset == 0:
            return

        print("ready to cache", cache_offset, self.phone_timings[cache_offset])
        print("offset before cache:", self.offset)
        self.cache_phones(cache_offset, self.phone_timings[cache_offset])
        print("offset after cache:", self.offset)

        return        

    def cache_previous_phones(self, offset):
        return self.cache_phones(offset, len(self.current_phones))

    def cache_phones(self, offset, up_to_char):
        print("Caching", offset, up_to_char)

        #current_phone_diff = len(self.current_transcript) - len(self.current_phones)
        #up_to_char = up_to_char - current_phone_diff
        #print("After diff:", up_to_char, current_phone_diff, self.current_transcript, "-", self.current_phones)

        phones_to_cache = self.current_phones[:up_to_char].strip()
        new_current_phones = self.current_phones[up_to_char:].strip()

        print("Splitting:", phones_to_cache, "-", new_current_phones)

        number_of_phones_to_cache = len(phones_to_cache.strip().split(' '))
        #self.previous_phone_timings.append(self.current_phone_timings[:number_of_phones_to_cache])
        self.current_phone_timings = self.current_phone_timings[number_of_phones_to_cache:]

        self.previous_phones += " " + phones_to_cache
        self.previous_phones = self.previous_phones.strip()
        self.current_phones = new_current_phones
        self.silence_time = 0
        #self.current_samples = np.empty(0, int)
        self.offset = offset
        self.last_cache_offset = offset
        #self.frame_offset = 0
        #self.speech_found = False
        self.total_speech_ms = 0

        print("Cached:", self.previous_phones, self.previous_phone_timings)

    def write(self, samples):
        self.current_samples = np.concatenate((self.current_samples, samples))

        self.process_new_samples()

    def store_phone_timings(self, new_transcript, offset):
        if len(new_transcript) > len(self.current_transcript):
            self.phone_timings[offset] = len(new_transcript)
            print("Phone timings:", self.phone_timings)

        self.current_transcript = new_transcript
 
    def process_new_samples(self):
        frame_duration = 20
        frame_n = int(8000 * frame_duration / 1000)

        while self.frame_offset + frame_n < len(self.current_samples):
            frame = self.current_samples[self.frame_offset:self.frame_offset+frame_n]     
            is_speech = self.vad.is_speech(frame, 8000)
            #print("Is speech:", is_speech)

            if not is_speech:
                print("Silence!")
                self.total_silence_ms += frame_duration
            elif is_speech and not self.speech_found:
                print("Speech started")
                self.total_silence_ms = 0
                self.speech_found = True
                self.offset = self.frame_offset
                if self.offset > frame_n:
                    self.offset = self.offset - frame_n # backtrack a bit   
                
            if is_speech:
                self.total_silence_ms = 0
                self.total_speech_ms += frame_duration

            if self.total_speech_ms >= 400:
                print("Enough speech, sending interim results")
                self.total_speech_ms = 0
                new_transcript = self.recognize_new_samples(self.current_samples[self.offset:self.frame_offset+frame_n])
                self.store_phone_timings(new_transcript, self.frame_offset+frame_n)

            if self.total_silence_ms >= 200 and self.speech_found:
                print("200ms silence detected, sending current samples and caching result", self.offset, self.frame_offset)
                new_transcript = self.recognize_new_samples(self.current_samples[self.offset:self.frame_offset+frame_n])
                self.store_phone_timings(new_transcript, self.frame_offset+frame_n)
                #self.offset = self.frame_offset+frame_n                
                self.cache_previous_phones(self.frame_offset+frame_n)
                self.speech_found = False

            self.cache_phones_if_ready(self.frame_offset+frame_n)

            self.frame_offset += frame_n

        return

    def silence(self):
        print("!! Silence timeout")
        self.frame_offset = len(self.current_samples)
        if self.offset < len(self.current_samples):
            self.recognize_new_samples(self.current_samples[self.offset:])
            self.cache_previous_phones(self.frame_offset)

    def write2(self, samples):
        #self.offset = len(self.current_samples)
        
        #self.samples = np.concatenate((self.samples, samples))
        #n = int(self.audio_config.frame_rate * (300 / 1000.0))
        #self.offset = 0
        #min_samples = int(self.audio_config.frame_rate * (1000 / 1000.0))

        #self.frame_offset = len(self.current_samples)
        self.current_samples = np.concatenate((self.current_samples, samples))
        
        frame_duration = 20
        #self.frame_offset = self.offset
        frame_n = int(8000 * frame_duration / 1000)
        print("frame_n is", frame_n, "offset", self.offset, "frame offset", self.frame_offset, "len current samples", len(self.current_samples))
        #self.total_silence_ms = 0
        #self.total_speech_ms = 0
        #self.samples_processed = False
        #self.speech_found = False

        while self.frame_offset + frame_n < len(self.current_samples):
            print("frame_n is", frame_n, "offset", self.offset, "frame offset", self.frame_offset, "len current samples", len(self.current_samples))
            frame = self.current_samples[self.frame_offset:self.frame_offset+frame_n]            
            is_speech = self.vad.is_speech(frame, 8000)
            print("Contains speech:", is_speech)
            if not is_speech:
                print("Silence")
                self.total_silence_ms += frame_duration
            elif not self.speech_found:
                print("Speech started")
                self.total_silence_ms = 0
                self.speech_found = True
            elif self.samples_processed:
                print("New speech detected, resetting")                
                self.offset = self.frame_offset
                if self.offset > frame_n:
                    self.offset = self.offset - frame_n # backtrack a bit                
                self.total_silence_ms = 0
                self.speech_found = True
                self.samples_processed = False

            if is_speech:
                self.total_speech_ms += frame_duration 
            
            if self.total_speech_ms >= 400:
                print("Enough speech, sending current samples")
                print("before send offset", self.offset, "frame offset", self.frame_offset, "len current samples", len(self.current_samples))
                self.total_speech_ms = 0
                self.recognize_new_samples(self.current_samples[self.offset:self.frame_offset+frame_n])
                #self.offset = self.frame_offset+frame_n
                
                # self.offset = self.frame_offset
                # if self.offset > frame_n:
                #     self.offset = self.offset - frame_n

            if self.total_silence_ms >= 200 and self.speech_found and not self.samples_processed:
                print("200ms silence detected, sending current samples")
                self.recognize_new_samples(samples[self.offset:self.frame_offset+frame_n])
                #self.offset = self.frame_offset+frame_n
                self.samples_processed = True
                self.total_speech_ms = 0
                print("Caching")
                self.cache_previous_phones()
                #return

            self.frame_offset += frame_n

        # if len(self.current_samples) < min_samples:
        #     frame = samples[offset:offset+min_samples]
        #     self.recognize_new_samples(frame)            
        #     offset += min_samples
        
        # while offset + n < len(samples):
        #     frame = samples[offset:offset+n]
        #     self.recognize_new_samples(frame)            
        #     offset += n
        

        #self.offset = frame_offset
        
        # if frame_offset < len(self.current_samples):
        #     print("Sending final short samples")
        #     frame = self.current_samples[frame_offset:]
        #     self.recognize_new_samples(frame)

        # self.offset = len(self.current_samples)
