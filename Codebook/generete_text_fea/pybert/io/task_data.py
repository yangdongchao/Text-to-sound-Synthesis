import random
import pandas as pd
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar
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
event_to_id = {label : i for i, label in enumerate(events_label)}
class TaskData(object):
    def __init__(self):
        pass
    def train_val_split(self,X, y,valid_size,stratify=False,shuffle=True,save = True,
                        seed = None,data_name = None,data_dir = None):
        pbar = ProgressBar(n_total=len(X),desc='bucket')
        logger.info('split raw data into train and valid')
        if stratify:
            num_classes = len(list(set(y)))
            train, valid = [], []
            bucket = [[] for _ in range(num_classes)]
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
                pbar(step=step)
            del X, y
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * valid_size)
                if shuffle:
                    random.seed(seed)
                    random.shuffle(bt)
                valid.extend(bt[:test_size])
                train.extend(bt[test_size:])
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        else:
            data = []
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y))
                pbar(step=step)
            del X, y
            N = len(data)
            test_size = int(N * valid_size)
            if shuffle:
                random.seed(seed)
                random.shuffle(data)
            valid = data[:test_size]
            train = data[test_size:]
            # 混洗train数据集
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        if save:
            train_path = data_dir / f"{data_name}.train.pkl"
            valid_path = data_dir / f"{data_name}.valid.pkl"
            save_pickle(data=train,file_path=train_path)
            save_pickle(data = valid,file_path=valid_path)
        return train, valid

    def read_data_caps(self,raw_data_path,preprocessor = None,is_train=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        targets, sentences = [], []
        data = pd.read_csv(raw_data_path,sep='\t',usecols=[0,1,2])
        for row in data.values:
            if is_train:
                target = [0]*527
                # print('tmp ',target)
                tmp = row[2]
                # print('target ',tmp)
                target_ls = tmp.split(',')
                # print('target_ls ',target_ls)
                flag = 1
                for t in target_ls:
                    id_ = event_to_id[t]
                    # print('id_ ',id_)
                    target[id_] = 1
                    flag = 0
                if flag:
                    print(row)
                    assert 1==2
            else:
                target = [-1]*527
            sentence = str(row[1])
            # print('sentence ',sentence)
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
            # assert 1==2
        return targets,sentences

    def read_data(self,raw_data_path,preprocessor = None,is_train=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        targets, sentences = [], []
        data = pd.read_csv(raw_data_path)
        for row in data.values:
            if is_train:
                target = row[2:]
            else:
                target = [-1,-1,-1,-1,-1,-1]
            sentence = str(row[1])
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
        return targets,sentences

