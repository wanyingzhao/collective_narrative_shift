""" this script takes a batch of tweets as input and return SRL description.
    The output is a list contains:
    - (ARG0, Verb, ARG1)
    - Verb
    - frame
    - frame_score
    - offset
        
"""
from transformer_srl import dataset_readers, models, predictors
from allennlp.models.archival import load_archive
from utils import *
import torch
import random
INTERNAL_BATCH_SIZE = 500


cuda_idx =  "2" #random.sample(['0','1', '2'],1)[0]
device = torch.device("cuda:"+cuda_idx if torch.cuda.is_available() else "cpu")
print(device)

################# load model###################
# Please download the pretrained model from
# https://www.dropbox.com/s/4tes6ypf2do0feb/srl_bert_base_conll2012.tar.gz
# and replace MODEL_PATH
MODEL_PATH = "/home/zhaowany/INCAS/incas-iu/workflow/scripts/transformer-srl/srl_bert_base_conll2012.tar.gz"
PREDICTOR = predictors.SrlTransformersPredictor.from_path(MODEL_PATH,  cuda_device= int(cuda_idx))

##############################################
def get_batch(ids, data_to_batch):
    """
      data_to_batch is a dictionary, where key is sent_id,
      and value is sentence
    """
    data_to_batch = [(sent_id, {"sentence": sent}) for sent_id, sent in zip(ids, data_to_batch)]
    batchs = [data_to_batch[idx: idx+INTERNAL_BATCH_SIZE] for idx in range(0, len(data_to_batch), INTERNAL_BATCH_SIZE)]
    return batchs

def get_SRL_description(batch_data, predictor=PREDICTOR):
  
    def _run_predictor_batch(batch_data):
        results = []

        sent_ids = [sent_id for sent_id, sentence in batch_data]
        sentences = [sentence for sent_id, sentence in batch_data]
        try:
            train_results = predictor.predict_batch_json(sentences)
            results += list(zip(sent_ids, train_results, sentences))
        except: # when batch prediction fails, try prediction sent by sent
            for sent_id, sent in batch_data:
                try:
                    sent_proc = sent["sentence"]
                    train_result = predictor.predict(sentence = sent_proc)
                except:
                    print(sent)
                    continue
                results.append((sent_id, train_result, sent))

        outputs = []
        for sent_id, output, sent in results:
            records = output['verbs']
            for r in records:
                narrative_triplet = get_narrative_triplet(r)
                if narrative_triplet:
                    outputs.append((sent_id, narrative_triplet, sent))
        return outputs
    
    return _run_predictor_batch(batch_data)
    

def predict_sentences(ids, sents):
    """
         input: - sentences, list of sentences(str)

         output: list of formated narratives (dictionary)
                 Example keys are (example record):
                  - "verb" : 'could',
                  - "narrative": '(ARG0, Verb, ARG1)',
                  - "frame": 'go.04'
                  - "frame_score": '0.10186545550823212,'
    """
                             
    # batch data
    batch_data = get_batch(ids, sents)

    preds= []

    # predict SRL description by batch
    for batch in batch_data:
        srl_sent = get_SRL_description(batch)
        preds +=  srl_sent

    outputs = df_format(preds)
    return outputs
