import os
import sys
import nltk
nltk.data.path.append("/mnt/data1/zjx/nltk_data/corpora/wordnet")

import json
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import tqdm
import ssl
import pickle
from collections import defaultdict

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 直接下载，nltk.download 会自动检查是否已存在
print(nltk.data)
for package in ['averaged_perceptron_tagger', 'wordnet', 'punkt', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        print(f"⚠️ Missing nltk package: {package}, please install manually")


# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


def combine_coco_captions(annotation_path):
    if not os.path.exists('%s/captions_%s2017.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2017.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2017.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2017.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}
    return all_caps 


def combine_coco_instances(annotation_path):
    if not os.path.exists('%s/instances_%s2017.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2017.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2017.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2017.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}
    return all_instances 


class CHAIR(object):

    def __init__(self, coco_path):
        self.imid_to_objects = defaultdict(list)
        self.coco_path = coco_path

        synonyms = synonyms_txt.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = []
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
        
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'
        
        self.get_annotations()

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):
        self.caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(self.caps) == len(self.eval_imids)

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def caption_to_words(self, caption):
        words = nltk.word_tokenize(caption.lower())
        tagged_sent = nltk.pos_tag(words)
        lemmas_sent = []
        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        words = lemmas_sent
    
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i) 
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict: 
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        words = double_words
    
        if ('toilet' in words) & ('seat' in words): 
            words = [word for word in words if word != 'seat']
    
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        id_to_name = {}
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write("\rGetting annotations for %d/%d segmentation masks" 
                              %(i, len(segment_annotations)))
            imid = annotation['image_id']
            node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
            self.imid_to_objects[imid].append(node_word)
        print("\n")

    def get_annotations_from_captions(self):
        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write('\rGetting annotations for %d/%d ground truth captions' 
                              %(i, len(coco_caps['annotations'])))
            imid = annotation['image_id']
            _, node_words, _, _ = self.caption_to_words(annotation['caption'])
            self.imid_to_objects[imid].extend(node_words)
        print("\n")

    def get_annotations(self):
        self.get_annotations_from_segments() 
        self.get_annotations_from_captions()
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_chair(self, cap_file, image_id_key, caption_key):
        self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)
        
        imid_to_objects = self.imid_to_objects
        caps = self.caps
        eval_imids = self.eval_imids
 
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        len_caps = 0.
        num_recall_gt_objects = 0.
        num_gt_objects = 0.

        output = {'sentences': []} 
        
        for i in tqdm.trange(len(caps)):
            cap = caps[i]
            imid = eval_imids[i]
    
            words, node_words, idxs, raw_words = self.caption_to_words(cap) 
 
            gt_objects = imid_to_objects[imid]
            cap_dict = {'image_id': imid, 
                        'caption': cap,
                        'mscoco_hallucinated_words': [],
                        'mscoco_gt_words': list(gt_objects),
                        'mscoco_generated_words': list(node_words),
                        'hallucination_idxs': [], 
                        'words': raw_words 
                        }

            cap_dict['metrics'] = {'CHAIRs': 0,
                                   'CHAIRi': 0,
                                   'Recall': 0,
                                   'Len': 0,
                                   }
 
            coco_word_count += len(node_words) 
            hallucinated = False
            recall_gt_objects = set()
            
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1 
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    recall_gt_objects.add(node_word)
    
            num_caps += 1
            len_caps += len(raw_words)
            if hallucinated:
               num_hallucinated_caps += 1
            
            num_gt_objects += len(gt_objects)
            num_recall_gt_objects += len(recall_gt_objects)
    
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words'])/float(len(words))
            
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)
   
            output['sentences'].append(cap_dict)
 
        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        recall = num_recall_gt_objects / num_gt_objects
        avg_len = (0.01*len_caps/num_caps)
    
        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRi': chair_i,
                                     'Recall': recall,
                                     'Len': avg_len,}
    
        return output 


def load_generated_captions(cap_file, image_id_key, caption_key):
    ext = os.path.splitext(cap_file)[-1]
    if ext == '.json':
        caps = json.load(open(cap_file))
    elif ext == '.jsonl':
        caps = [json.loads(s) for s in open(cap_file)]
    else:
        raise ValueError(f'Unsupported extension {ext} for cap_file: {cap_file}')

    imids = [obj[image_id_key] for obj in caps]
    caps = [obj[caption_key] for obj in caps]
       
    return caps, imids


def save_hallucinated_words(cap_file, cap_dict): 
    folder = os.path.dirname(cap_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(cap_file, 'w', encoding='utf-8') as f:
        json.dump(cap_dict, f, indent=2, ensure_ascii=False)


def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    
    for k, v in sentence_metrics.items():
        k_str = str(k).ljust(10)
        v_str = f'{v * 100:.01f}'
        print(k_str, v_str, sep=': ')


def CHAIR_SCORE(cap_file, image_id_key='image_id', caption_key='caption', 
                   coco_path='coco_annotations', cache='chair.pkl', save_path=''):
    """
    CHAIR 评估函数
    
    Args:
        cap_file (str): JSON 或 JSONL 文件路径，包含图片ID和对应的标题
        image_id_key (str): cap_file 中存储图片ID的字段名，默认为 'image_id'
        caption_key (str): cap_file 中存储标题的字段名，默认为 'caption'
        coco_path (str): COCO 标注文件路径，默认为 'coco_annotations'
        cache (str): 缓存的 CHAIR 评估器文件路径，默认为 'chair.pkl'
        save_path (str): 保存评估结果的 JSON 文件路径。如果为空，不保存
    
    Returns:
        dict: 包含详细评估结果的字典
    
    Example:
        result = evaluate_chair(
            cap_file='/path/to/captions.jsonl',
            image_id_key='image_id',
            caption_key='caption',
            coco_path='/path/to/coco',
            cache='/path/to/chair.pkl',
            save_path='/path/to/results.json'
        )
    """
    
    if cache and os.path.exists(cache):
        evaluator = pickle.load(open(cache, 'rb'))
        print(f"Loaded evaluator from cache: {cache}")
    else:
        print(f"Cache not found, building evaluator from scratch...")
        evaluator = CHAIR(coco_path)
        if cache:
            pickle.dump(evaluator, open(cache, 'wb'))
            print(f"Cached evaluator to: {cache}")

    cap_dict = evaluator.compute_chair(cap_file, image_id_key, caption_key) 
    
    print_metrics(cap_dict)
    
    if save_path:
        save_hallucinated_words(save_path, cap_dict)
        print(f"Results saved to: {save_path}")
    
    return cap_dict


# if __name__ == '__main__':
#     # 使用示例
#     result = CHAIR_SCORE(
#         cap_file='/mnt/data1/zjx/project/alw/eval/temp/dola.jsonl',
#         image_id_key='image_id',
#         caption_key='caption',
#         coco_path='/mnt/data1/zjx/project/alw/data/raw_data/chair/data/coco_path',
#         cache='/mnt/data1/zjx/project/alw/eval/chair_2017.pkl',
#         save_path='/mnt/data1/zjx/project/alw/eval/results/test/dola.json'
#     )