import csv
import torch
import numpy as np
from ..common.tools import load_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid   = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeature(object):
    '''
    A single set of features of data.
    '''
    def __init__(self,input_ids,input_mask,segment_ids,label_id,input_len):
        self.input_ids   = input_ids
        self.input_mask  = input_mask
        self.segment_ids = segment_ids
        self.label_id    = label_id
        self.input_len = input_len

class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,vocab_path,do_lower_case):
        self.tokenizer = BertTokenizer(vocab_path,do_lower_case)

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self,lines):
        return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        events_label = ['Speech', 'man_speaking', 'woman_speaking', 'kid_speaking', 'Conversation', 'monologue', 'Babbling',
                'Speech_synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell', 'Battle_cry', 'Children_shouting',
                'Screaming', 'Whispering', 'Laughter', 'Baby_laughter', 'Giggle', 'Snicker', 'Belly_laugh', 
                'chortle', 'sobbing', 'infant_cry', 'Whimper', 'moan', 'Sigh', 'Singing', 'Choir', 'Yodeling', 
                'Chant', 'Mantra', 'Male_singing', 'Female_singing', 'Child_singing', 'Synthetic_singing', 
                'Rapping', 'Humming', 'Groan', 'Grunt', 'Whistling', 'Breathing', 'Wheeze', 'Snoring', 
                'Gasp', 'Pant', 'Snort', 'Cough', 'Throat_clearing', 'Sneeze', 'Sniff', 'Run', 'Shuffle', 
                'footsteps', 'mastication', 'Biting', 'Gargling', 'Stomach_rumble', 'eructation', 'Hiccup', 
                'Fart', 'Hands', 'Finger_snapping', 'Clapping', 'heartbeat', 'Heart_murmur', 'Cheering', 
                'Applause', 'Chatter', 'Crowd', 'speech_babble', 'Children_playing', 'Animal', 'pets', 
                'Dog', 'Bark', 'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper_(dog)', 'Cat', 'Purr', 
                'Meow', 'Hiss', 'Caterwaul', 'working_animals', 'Horse', 'Clip-clop', 'whinny', 'bovinae', 
                'Moo', 'Cowbell', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl', 'rooster', 'Cluck', 
                'cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack', 'Goose', 'Honk', 'Wild_animals', 
                'tigers)', 'Roar', 'Bird', 'bird_song', 'tweet', 'Squawk', 'dove', 'Coo', 'Crow', 'Caw', 
                'Owl', 'Hoot', 'flapping_wings', 'wolves', 'mice', 'Mouse', 'Patter', 'Insect', 'Cricket', 
                'Mosquito', 'housefly', 'Buzz', 'etc.', 'Frog', 'Croak', 'Snake', 'Rattle', 'Whale_vocalization', 
                'Music', 'Musical_instrument', 'Plucked_string_instrument', 'Guitar', 'Electric_guitar', 
                'Bass_guitar', 'Acoustic_guitar', 'slide_guitar', 'Tapping_(guitar_technique)', 'Strum', 
                'Banjo', 'Sitar', 'Mandolin', 'Zither', 'Ukulele', 'Keyboard_(musical)', 'Piano', 
                'Electric_piano', 'Organ', 'Electronic_organ', 'Hammond_organ', 'Synthesizer', 
                'Sampler', 'Harpsichord', 'Percussion', 'Drum_kit', 'Drum_machine', 'Drum', 'Snare_drum', 
                'Rimshot', 'Drum_roll', 'Bass_drum', 'Timpani', 'Tabla', 'Cymbal', 'Hi-hat', 'Wood_block', 
                'Tambourine', 'Rattle_(instrument)', 'Maraca', 'Gong', 'Tubular_bells', 'Mallet_percussion', 
                'xylophone', 'Glockenspiel', 'Vibraphone', 'Steelpan', 'Orchestra', 'Brass_instrument', 
                'French_horn', 'Trumpet', 'Trombone', 'Bowed_string_instrument', 'String_section', 'fiddle', 
                'Pizzicato', 'Cello', 'Double_bass', 'woodwind_instrument', 'Flute', 'Saxophone', 'Clarinet', 
                'Harp', 'Bell', 'Church_bell', 'Jingle_bell', 'Bicycle_bell', 'Tuning_fork', 'Chime', 'Wind_chime', 
                'Change_ringing_(campanology)', 'Harmonica', 'Accordion', 'Bagpipes', 'Didgeridoo', 'Shofar', 
                'Theremin', 'Singing_bowl', 'Scratching_(performance_technique)', 'Pop_music', 'Hip_hop_music', 
                'Beatboxing', 'Rock_music', 'Heavy_metal', 'Punk_rock', 'Grunge', 'Progressive_rock', 
                'Rock_and_roll', 'Psychedelic_rock', 'Rhythm_and_blues', 'Soul_music', 'Reggae', 'Country', 
                'Swing_music', 'Bluegrass', 'Funk', 'Folk_music', 'Middle_Eastern_music', 'Jazz', 'Disco', 
                'Classical_music', 'Opera', 'Electronic_music', 'House_music', 'Techno', 'Dubstep', 'Drum_and_bass', 
                'Electronica', 'Electronic_dance_music', 'Ambient_music', 'Trance_music', 'Music_of_Latin_America', 
                'Salsa_music', 'Flamenco', 'Blues', 'Music_for_children', 'New-age_music', 'Vocal_music', 
                'A_capella', 'Music_of_Africa', 'Afrobeat', 'Christian_music', 'Gospel_music', 'Music_of_Asia', 
                'Carnatic_music', 'Music_of_Bollywood', 'Ska', 'Traditional_music', 'Independent_music', 'Song', 
                'Background_music', 'Theme_music', 'Jingle_(music)', 'Soundtrack_music', 'Lullaby', 
                'Video_game_music', 'Christmas_music', 'Dance_music', 'Wedding_music', 'Happy_music', 
                'Funny_music', 'Sad_music', 'Tender_music', 'Exciting_music', 'Angry_music', 'Scary_music', 
                'Wind', 'Rustling_leaves', 'Wind_noise_(microphone)', 'Thunderstorm', 'Thunder', 'Water', 
                'Rain', 'Raindrop', 'Rain_on_surface', 'Stream', 'Waterfall', 'Ocean', 'surf', 'Steam', 
                'Gurgling', 'Fire', 'Crackle', 'Vehicle', 'Water_vehicle', 'sailing_ship', 'kayak', 'speedboat', 
                'Ship', 'Motor_vehicle_(road)', 'Car', 'honking', 'Toot', 'Car_alarm', 'electric_windows', 
                'Skidding', 'Tire_squeal', 'Car_passing_by', 'auto_racing', 'Truck', 'Air_brake', 'truck_horn', 
                'Reversing_beeps', 'ice_cream_van', 'Bus', 'Emergency_vehicle', 'Police_car_(siren)', 
                'Ambulance_(siren)', 'fire_truck_(siren)', 'Motorcycle', 'roadway_noise', 'Rail_transport', 
                'Train', 'Train_whistle', 'Train_horn', 'train_wagon', 'Train_wheels_squealing', 'underground', 
                'Aircraft', 'Aircraft_engine', 'Jet_engine', 'airscrew', 'Helicopter', 'airplane', 'Bicycle', 
                'Skateboard', 'Engine', 'Light_engine_(high_frequency)', "dentist's_drill", 'Lawn_mower', 
                'Chainsaw', 'Medium_engine_(mid_frequency)', 'Heavy_engine_(low_frequency)', 'Engine_knocking', 
                'Engine_starting', 'Idling', 'vroom', 'Door', 'Doorbell', 'Ding-dong', 'Sliding_door', 'Slam', 
                'Knock', 'Tap', 'Squeak', 'Cupboard_open_or_close', 'Drawer_open_or_close', 'and_pans', 
                'silverware', 'Chopping_(food)', 'Frying_(food)', 'Microwave_oven', 'Blender', 'faucet', 
                'Sink_(filling_or_washing)', 'Bathtub_(filling_or_washing)', 'Hair_dryer', 'Toilet_flush', 
                'Toothbrush', 'Electric_toothbrush', 'Vacuum_cleaner', 'Zipper_(clothing)', 'Keys_jangling', 
                'Coin_(dropping)', 'Scissors', 'electric_razor', 'Shuffling_cards', 'Typing', 'Typewriter', 
                'Computer_keyboard', 'Writing', 'Alarm', 'Telephone', 'Telephone_bell_ringing', 'Ringtone', 
                'DTMF', 'Dial_tone', 'Busy_signal', 'Alarm_clock', 'Siren', 'Civil_defense_siren', 'Buzzer', 
                'smoke_alarm', 'Fire_alarm', 'Foghorn', 'Whistle', 'Steam_whistle', 'Mechanisms', 'pawl', 
                'Clock', 'Tick', 'Tick-tock', 'Gears', 'Pulleys', 'Sewing_machine', 'Mechanical_fan', 
                'Air_conditioning', 'Cash_register', 'Printer', 'Camera', 'Single-lens_reflex_camera', 
                'Tools', 'Hammer', 'Jackhammer', 'Sawing', 'Filing_(rasp)', 'Sanding', 'Power_tool', 
                'Drill', 'Explosion', 'gunfire', 'Machine_gun', 'Fusillade', 'Artillery_fire', 'Cap_gun', 
                'Fireworks', 'Firecracker', 'pop', 'Eruption', 'Boom', 'Wood', 'Chop', 'Splinter', 'Crack', 
                'Glass', 'clink', 'Shatter', 'Liquid', 'splatter', 'Slosh', 'Squish', 'Drip', 'Pour', 'dribble', 
                'Gush', 'Fill_(with_liquid)', 'Spray', 'Pump_(liquid)', 'Stir', 'Boiling', 'Sonar', 'Arrow', 
                'swish', 'thud', 'Thunk', 'Electronic_tuner', 'Effects_unit', 'Chorus_effect', 'Basketball_bounce', 
                'Bang', 'smack', 'thwack', 'crash', 'Breaking', 'Bouncing', 'Whip', 'Flap', 'Scratch', 'Scrape', 
                'Rub', 'Roll', 'Crushing', 'crinkling', 'Tearing', 'bleep', 'Ping', 'Ding', 'Clang', 'Squeal', 
                'Creak', 'Rustle', 'Whir', 'Clatter', 'Sizzle', 'Clicking', 'Clickety-clack', 'Rumble', 'Plop', 
                'tinkle', 'Hum', 'Zing', 'Boing', 'Crunch', 'Silence', 'Sine_wave', 'Harmonic', 'Chirp_tone', 
                'Sound_effect', 'Pulse', 'small_room', 'large_room_or_hall', 'public_space', 'urban_or_manmade', 
                'rural_or_natural', 'Reverberation', 'Echo', 'Noise', 'Environmental_noise', 'Static', 'Mains_hum', 
                'Distortion', 'Sidetone', 'Cacophony', 'White_noise', 'Pink_noise', 'Throbbing', 'Vibration', 
                'Television', 'Radio', 'Field_recording']
        return events_label

    @classmethod
    def read_data(cls, input_file,quotechar = None):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self,lines,example_type,cached_examples_file):
        '''
        Creates examples for data
        '''
        pbar = ProgressBar(n_total = len(lines),desc='create examples')
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i,line in enumerate(lines):
                guid = '%s-%d'%(example_type,i)
                text_a = line[0]
                label = line[1]
                if isinstance(label,str):
                    label = [np.float(x) for x in label.split(",")]
                else:
                    label = [np.float(x) for x in list(label)]
                text_b = None
                example = InputExample(guid = guid,text_a = text_a,text_b=text_b,label= label)
                examples.append(example)
                pbar(step=i)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_features(self,examples,max_seq_len,cached_features_file):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        '''
        pbar = ProgressBar(n_total=len(examples),desc='create features')
        if cached_features_file.exists():
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id,example in enumerate(examples):
                tokens_a = self.tokenizer.tokenize(example.text_a)
                tokens_b = None
                label_id = example.label

                if example.text_b:
                    tokens_b = self.tokenizer.tokenize(example.text_b)
                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP] with "- 3"
                    self.truncate_seq_pair(tokens_a,tokens_b,max_length = max_seq_len - 3)
                else:
                    # Account for [CLS] and [SEP] with '-2'
                    if len(tokens_a) > max_seq_len - 2:
                        tokens_a = tokens_a[:max_seq_len - 2]
                tokens = ['[CLS]'] + tokens_a + ['[SEP]']
                segment_ids = [0] * len(tokens)
                if tokens_b:
                    tokens += tokens_b + ['[SEP]']
                    segment_ids += [1] * (len(tokens_b) + 1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                padding = [0] * (max_seq_len - len(input_ids))
                input_len = len(input_ids)

                input_ids   += padding
                input_mask  += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len

                if ex_id < 2:
                    logger.info("*** Example ***")
                    logger.info(f"guid: {example.guid}" % ())
                    logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                    logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                    logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")

                feature = InputFeature(input_ids = input_ids,
                                       input_mask = input_mask,
                                       segment_ids = segment_ids,
                                       label_id = label_id,
                                       input_len = input_len)
                features.append(feature)
                pbar(step=ex_id)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self,features,is_sorted = False):
        # Convert to Tensors and build dataset
        if is_sorted:
            logger.info("sorted data by th length of input")
            features = sorted(features,key=lambda x:x.input_len,reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features],dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_input_lens)
        return dataset

