import collections
import csv
import logging
import os
import random
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchvision
logger = logging.getLogger(f'main.{__name__}')
class_dict = ['Railroad car or train wagon', 'Train horn', 'Rail transport', 'Train', 'Clickety-clack', 'Speech', 'Narration or monologue', 'Female speech or woman speaking', 'Male speech or man speaking', 'Boat or Water vehicle', 'Vehicle', 'Music', 'Sound effect', 'Motor vehicle road', 'Car', 'Car passing by', 'Heavy engine low frequency', 'Run', 'Bass guitar', 'Guitar', 'Musical instrument', 'Plucked string instrument', 'Gurgling', 'Waterfall', 'Stream', 'Female singing', 'Dance music', 'Roll', 'Bus', 'Clatter', 'Truck', 'Drum', 'Drum kit', 'Bass drum', 'Acoustic guitar', 'Singing', 'Music of Africa', 'Hip hop music', 'Male singing', 'Aircraft', 'Bird', 'Bird vocalization or bird call or bird song', 'Chirp or tweet', 'Drum machine', 'Wind', 'Wind noise microphone', 'Sliding door', 'Engine', 'Medium engine mid frequency', 'Engine starting', 'Brass instrument', 'Trombone', 'Electronic music', 'Ambient music', 'Synthesizer', 'Inside or large room or hall', 'Orchestra', 'Inside or small room', 'Tender music', 'Choir', 'Reggae', 'Fixed-wing aircraft or airplane', 'Belly laugh', 'Child speech or kid speaking', 'Cat', 'Domestic animals or pets', 'Animal', 'Electric guitar', 'Strum', 'Rhythm and blues', 'Outside or rural or natural', 'Independent music', 'Siren', 'Dog', 'Ship', 'Country', 'Wood block', 'Skidding', 'Background music', 'Heavy metal', 'Gasp', 'Salsa music', 'Dubstep', 'Rock music', 'Chuckle or chortle', 'Crowd', 'Basketball bounce', 'Radio', 'Electronic tuner', 'Reverberation', 'Rustle', 'Sizzle', 'Tick-tock', 'Tick', 'Happy music', 'Speech synthesizer', 'Classical music', 'Idling', 'Accelerating or revving or vroom', 'Bow-wow', 'Video game music', 'Motorboat or speedboat', 'Babbling', 'Effects unit', 'Traditional music', 'Vocal music', 'Harpsichord', 'Pop music', 'Writing', 'Vehicle horn or car horn or honking', 'Horse', 'Clip-clop', 'Soul music', 'Keyboard musical', 'Piano', 'Neigh or whinny', 'Crow', 'Jingle music', 'Chainsaw', 'Lawn mower', 'Zither', 'Jazz', 'Bowed string instrument', 'Outside or urban or manmade', 'Sad music', 'Mechanical fan', 'Caterwaul', 'Music of Latin America', 'Violin or fiddle', 'Sonar', 'Vibration', 'Yip', 'Whimper dog', 'Applause', 'Water tap or faucet', 'Cheering', 'Inside or public space', 'Bee or wasp or etc.', 'Insect', 'Fly or housefly', 'Christmas music', 'Water', 'Marimba or xylophone', 'Glockenspiel', 'Mallet percussion', 'Pizzicato', 'Ukulele', 'Fire alarm', 'Techno', 'Hammer', 'Rattle', 'Gunshot or gunfire', 'Machine gun', 'Exciting music', 'Whispering', 'Firecracker', 'Flamenco', 'Rain', 'Raindrop', 'Soundtrack music', 'New-age music', 'Tearing', 'Printer', 'Whoop', 'Whoosh or swoosh or swish', 'Harp', 'Drill', 'Tools', 'Power tool', 'Harmonica', 'Theme music', 'Accordion', 'Scary music', 'Bathtub filling or washing', 'Funk', 'Tabla', 'Percussion', 'House music', 'Duck', 'Electronic dance music', 'Television', 'Child singing', 'Single-lens reflex camera', 'Electronica', 'Beatboxing', 'Race car or auto racing', 'Thunk', 'Walk or footsteps', 'Purr', 'Middle Eastern music', 'Environmental noise', 'Mandolin', 'Chant', 'Fire engine or fire truck siren', 'Emergency vehicle', 'Hair dryer', 'Trance music', 'Disco', 'Steam', 'Hiss', 'Ambulance siren', 'Conversation', 'Goat', 'Rapping', 'Punk rock', 'Motorcycle', 'Coo', 'Pigeon or dove', 'Cymbal', 'Christian music', 'Hi-hat', 'Snare drum', 'Didgeridoo', 'Snoring', 'Sewing machine', 'Gong', 'Burst or pop', 'Laughter', 'Bark', 'Snicker', 'Rimshot', 'Flute', 'Sampler', 'Sheep', 'Bleat', 'Air horn or truck horn', 'Slap or smack', 'Whack or thwack', 'Honk', 'Bagpipes', 'Wind instrument or woodwind instrument', 'Electric piano', 'Smash or crash', 'Helicopter', 'Timpani', 'Telephone', 'Mantra', 'Civil defense siren', 'Spray', 'Song', 'Air brake', 'Opera', 'Skateboard', 'Folk music', 'Silence', 'Music of Bollywood', 'Music for children', 'Jet engine', 'Tire squeal', 'Scratching performance technique', 'Toot', 'Traffic noise or roadway noise', 'Synthetic singing', 'Fowl', 'Cluck', 'Wind chime', 'Chime', 'Shuffling cards', 'Crunch', 'Crumpling or crinkling', 'Sink filling or washing', 'Boom', 'Chatter', 'Whistle', 'Rock and roll', 'Whistling', 'Trumpet', 'Slosh', 'Cello', 'Double bass', 'Rowboat or canoe or kayak', 'Chicken or rooster', 'Toilet flush', 'Quack', 'Blues', 'Bird flight or flapping wings', 'Baby cry or infant cry', 'Goose', 'Jingle bell', 'Whimper', 'Tapping guitar technique', 'Hubbub or speech noise or speech babble', 'Lullaby', 'Psychedelic rock', 'Saxophone', 'Banjo', 'Snort', 'Funny music', 'Crowing or cock-a-doodle-doo', 'Mosquito', 'Grunge', 'Steelpan', 'Door', 'Police car siren', 'Harmonic', 'Church bell', 'Electric shaver or electric razor', 'Tuning fork', 'Electronic organ', 'Gospel music', 'Drum roll', 'Plop', 'Rain on surface', 'Explosion', 'Fireworks', 'Stir', 'Frying food', 'Howl', 'Flap', 'Cowbell', 'Music of Asia', 'Vacuum cleaner', 'Fire', 'Wedding music', 'Pink noise', 'Livestock or farm animals or working animals', 'Distortion', 'Steel guitar or slide guitar', 'Ska', 'String section', 'Children playing', 'Ringtone', 'Crying or sobbing', 'Progressive rock', 'Telephone dialing or DTMF', 'Cash register', 'Telephone bell ringing', 'Ocean', 'Drum and bass', 'White noise', 'Sailboat or sailing ship', 'Waves or surf', 'Tap', 'Slam', 'Breaking', 'Clapping', 'A capella', 'Jingle or tinkle', 'Meow', 'Screaming', 'Crackle', 'Artillery fire', 'Pump liquid', 'Chop', 'Turkey', 'Vibraphone', 'Gobble', 'Propeller or airscrew', 'Engine knocking', 'Theremin', 'French horn', 'Biting', 'Thump or thud', 'Clarinet', 'Buzz', 'Shatter', 'Burping or eructation', 'Maraca', 'Beep or bleep', 'Bluegrass', 'Chink or clink', 'Afrobeat', 'Drip', 'Field recording', 'Hum', 'Ratchet or pawl', 'Mechanisms', 'Wild animals', 'Change ringing campanology', 'Shuffle', 'Fart', 'Gears', 'Fusillade', 'Bell', 'Whale vocalization', 'Blender', 'Groan', 'Angry music', 'Breathing', 'Buzzer', 'Wood', 'Trickle or dribble', 'Sneeze', 'Hands', 'Heart sounds or heartbeat', 'Coin dropping', 'Subway or metro or underground', 'Grunt', 'Roaring cats lions or tigers', 'Hammond organ', 'Canidae or dogs or wolves', 'Growling', 'Hiccup', 'Scratch', 'Bicycle', 'Croak', 'Frog', 'Owl', 'Throbbing', 'Chewing or mastication', 'Oink', 'Rub', 'Giggle', 'Gush', 'Filing rasp', 'Cacophony', 'Train wheels squealing', 'Cough', 'Organ', 'Squish', 'Clock', 'Arrow', "Dental drill or dentist's drill", 'Thunderstorm', 'Thunder', 'Microwave oven', 'Children shouting', 'Cattle or bovinae', 'Moo', 'Computer keyboard', 'Typing', 'Camera', 'Bang', 'Reversing beeps', 'Alarm clock', 'Alarm', 'Aircraft engine', 'Doorbell', 'Humming', 'Static', 'Electric toothbrush', 'Bellow', 'Eruption', 'Rustling leaves', 'Snake', 'Shout', 'Clicking', 'Keys jangling', 'Dishes or pots or and pans', 'Shofar', 'Ding', 'Steam whistle', 'Rattle instrument', 'Splash or splatter', 'Cricket', 'Train whistle', 'Cap gun', 'Knock', 'Air conditioning', 'Heart murmur', 'Patter', 'Jackhammer', 'Liquid', 'Singing bowl', 'Wail or moan', 'Carnatic music', 'Sitar', 'Fill with liquid', 'Echo', 'Tubular bells', 'Ding-dong', 'Pig', 'Sine wave', 'Ping', 'Battle cry', 'Swing music', 'Smoke detector or smoke alarm', 'Cutlery or silverware', 'Boiling', 'Power windows or electric windows', 'Drawer open or close', 'Light engine high frequency', 'Scrape', 'Throat clearing', 'Yodeling', 'Caw', 'Squeal', 'Sniff', 'Typewriter', 'Pour', 'Busy signal', 'Foghorn', 'Ice cream truck or ice cream van', 'Mouse', 'Rumble', 'Yell', 'Glass', 'Zipper clothing', 'Rodents or rats or mice', 'Whip', 'Roar', 'Sawing', 'Chopping food', 'Whir', 'Chirp tone', 'Tambourine', 'Dial tone', 'Baby laughter', 'Chorus effect', 'Boing', 'Car alarm', 'Mains hum', 'Bicycle bell', 'Wheeze', 'Sigh', 'Cupboard open or close', 'Noise', 'Pulleys', 'Pant', 'Clang', 'Sidetone', 'Stomach rumble', 'Sanding', 'Splinter', 'Zing', 'Toothbrush', 'Crack', 'Squeak', 'Bouncing', 'Crushing', 'Pulse', 'Scissors', 'Hoot', 'Creak', 'Finger snapping', 'Squawk', 'Gargling']
class_labels = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code3/SpecVQGAN/specvqgan/modules/losses/vggishish/data/unbalanced_train_exist_part.tsv', sep='\t',usecols=[0,1])
filename = class_labels['filename']
event_label = class_labels['event_label'] 
filename_ls = []
event_label_ls = []
for name in filename:
    filename_ls.append(name)
for eb in event_label:
    event_label_ls.append(eb)

id_to_class = {i: label for i,label in enumerate(class_dict)}
class_to_id = {label: i for i,label in id_to_class.items()}
filename_to_events = {}
for i in range(len(filename_ls)):
    filename_to_events[filename_ls[i]] = event_label_ls[i]

class Audioset(torch.utils.data.Dataset):
    def __init__(self, split, specs_dir, transforms=None, splits_path='./data', meta_path='./data/unbalanced_train_exist_part.tsv'):
        super().__init__()
        self.split = split
        self.specs_dir = specs_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.meta_path = meta_path
        # self.class_to_id, self.id_to_class, self.filename_to_events = read_tsv(meta_path)
        split_clip_ids_path = os.path.join(splits_path, f'audioset_{split}.txt') # it must exist
        clip_ids_with_timestamp = open(split_clip_ids_path).read().splitlines()
        clip_paths = [os.path.join(specs_dir, v + '_mel.npy') for v in clip_ids_with_timestamp]
        self.dataset = clip_paths
        # self.dataset = clip_paths[:10000]  # overfit one batch

        # 'zyTX_1BXKDE_16000_26000'[:11] -> 'zyTX_1BXKDE'
        # vid_classes = [self.video2target[Path(path).stem[:11]] for path in self.dataset]
        # class2count = collections.Counter(vid_classes)
        # self.class_counts = torch.tensor([class2count[cls] for cls in range(len(class2count))])
        # self.sample_weights = [len(self.dataset) / class2count[self.video2target[Path(path).stem[:11]]] for path in self.dataset]

    def __getitem__(self, idx):
        item = {}

        spec_path = self.dataset[idx]
        # print('spec_path ',spec_path)
        # # 'zyTX_1BXKDE_16000_26000' -> 'zyTX_1BXKDE'
        # print(Path(spec_path).stem)
        video_name = Path(spec_path).stem[:-4]
        # print(video_name)
        # assert 1==2
        item['input'] = np.load(spec_path)
        item['input_path'] = spec_path

        # if self.split in ['train', 'valid']:
        # item['target'] = self.video2target[video_name]
        #item['label'] = self.target2label[item['target']]
        tmp_target = np.zeros(527)
        tmp_events = filename_to_events[video_name]
        # print(tmp_events)
        tmp_ls = tmp_events.split(',')
        for l in tmp_ls:
            tmp_target[class_to_id[l]] = 1.0
        
        item['target'] = tmp_target
        # print('item ', item)
        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def __len__(self):
        return len(self.dataset)



class VGGSound(torch.utils.data.Dataset):
    def __init__(self, split, specs_dir, transforms=None, splits_path='./data', meta_path='./data/vggsound.csv'):
        super().__init__()
        self.split = split
        self.specs_dir = specs_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.meta_path = meta_path

        vggsound_meta = list(csv.reader(open(meta_path), quotechar='"'))
        # print('vggsound_meta ',vggsound_meta)
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        self.label2target = {label: target for target, label in enumerate(unique_classes)}
        self.target2label = {target: label for label, target in self.label2target.items()}
        self.video2target = {row[0]: self.label2target[row[2]] for row in vggsound_meta}
        split_clip_ids_path = os.path.join(splits_path, f'vggsound_{split}.txt')
        if not os.path.exists(split_clip_ids_path):
            self.make_split_files()
        clip_ids_with_timestamp = open(split_clip_ids_path).read().splitlines()
        clip_paths = [os.path.join(specs_dir, v + '_mel.npy') for v in clip_ids_with_timestamp]
        self.dataset = clip_paths
        # self.dataset = clip_paths[:10000]  # overfit one batch

        # 'zyTX_1BXKDE_16000_26000'[:11] -> 'zyTX_1BXKDE'
        vid_classes = [self.video2target[Path(path).stem[:11]] for path in self.dataset]
        class2count = collections.Counter(vid_classes)
        self.class_counts = torch.tensor([class2count[cls] for cls in range(len(class2count))])
        # self.sample_weights = [len(self.dataset) / class2count[self.video2target[Path(path).stem[:11]]] for path in self.dataset]

    def __getitem__(self, idx):
        item = {}

        spec_path = self.dataset[idx]
        # 'zyTX_1BXKDE_16000_26000' -> 'zyTX_1BXKDE'
        video_name = Path(spec_path).stem[:11]

        item['input'] = np.load(spec_path)
        item['input_path'] = spec_path

        # if self.split in ['train', 'valid']:
        item['target'] = self.video2target[video_name]
        item['label'] = self.target2label[item['target']]
        if self.transforms is not None:
            item = self.transforms(item)

        return item

    def __len__(self):
        return len(self.dataset)

    def make_split_files(self):
        random.seed(1337)
        logger.info(f'The split files do not exist @ {self.splits_path}. Calculating the new ones.')
        # The downloaded videos (some went missing on YouTube and no longer available)
        available_vid_paths = sorted(glob(os.path.join(self.specs_dir, '*_mel.npy')))
        logger.info(f'The number of clips available after download: {len(available_vid_paths)}')

        # original (full) train and test sets
        vggsound_meta = list(csv.reader(open(self.meta_path), quotechar='"'))
        train_vids = {row[0] for row in vggsound_meta if row[3] == 'train'}
        test_vids = {row[0] for row in vggsound_meta if row[3] == 'test'}
        logger.info(f'The number of videos in vggsound train set: {len(train_vids)}')
        logger.info(f'The number of videos in vggsound test set: {len(test_vids)}')

        # class counts in test set. We would like to have the same distribution in valid
        unique_classes = sorted(list(set(row[2] for row in vggsound_meta)))
        label2target = {label: target for target, label in enumerate(unique_classes)}
        video2target = {row[0]: label2target[row[2]] for row in vggsound_meta}
        test_vid_classes = [video2target[vid] for vid in test_vids]
        test_target2count = collections.Counter(test_vid_classes)

        # now given the counts from test set, sample the same count for validation and the rest leave in train
        train_vids_wo_valid, valid_vids = set(), set()
        for target, label in enumerate(label2target.keys()):
            class_train_vids = [vid for vid in train_vids if video2target[vid] == target]
            random.shuffle(class_train_vids)
            count = test_target2count[target]
            valid_vids.update(class_train_vids[:count])
            train_vids_wo_valid.update(class_train_vids[count:])

        # make file with a list of available test videos (each video should contain timestamps as well)
        train_i = valid_i = test_i = 0
        with open(os.path.join(self.splits_path, 'vggsound_train.txt'), 'w') as train_file, \
             open(os.path.join(self.splits_path, 'vggsound_valid.txt'), 'w') as valid_file, \
             open(os.path.join(self.splits_path, 'vggsound_test.txt'), 'w') as test_file:
            for path in available_vid_paths:
                path = path.replace('_mel.npy', '')
                vid_name = Path(path).name
                # 'zyTX_1BXKDE_16000_26000'[:11] -> 'zyTX_1BXKDE'
                if vid_name[:11] in train_vids_wo_valid:
                    train_file.write(vid_name + '\n')
                    train_i += 1
                elif vid_name[:11] in valid_vids:
                    valid_file.write(vid_name + '\n')
                    valid_i += 1
                elif vid_name[:11] in test_vids:
                    test_file.write(vid_name + '\n')
                    test_i += 1
                else:
                    raise Exception(f'Clip {vid_name} is neither in train, valid nor test. Strange.')

        logger.info(f'Put {train_i} clips to the train set and saved it to ./data/vggsound_train.txt')
        logger.info(f'Put {valid_i} clips to the valid set and saved it to ./data/vggsound_valid.txt')
        logger.info(f'Put {test_i} clips to the test set and saved it to ./data/vggsound_test.txt')


if __name__ == '__main__':
    from transforms import Crop, StandardNormalizeAudio, ToTensor
    specs_path = '/home/nvme/data/vggsound/features/melspec_10s_22050hz/'

    transforms = torchvision.transforms.transforms.Compose([
        StandardNormalizeAudio(specs_path),
        ToTensor(),
        Crop([80, 848]),
    ])

    datasets = {
        'train': VGGSound('train', specs_path, transforms),
        'valid': VGGSound('valid', specs_path, transforms),
        'test': VGGSound('test', specs_path, transforms),
    }

    print(datasets['train'][0])
    print(datasets['valid'][0])
    print(datasets['test'][0])

    print(datasets['train'].class_counts)
    print(datasets['valid'].class_counts)
    print(datasets['test'].class_counts)
