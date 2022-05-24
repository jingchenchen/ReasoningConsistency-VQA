from __future__ import generator_stop

import threading
import queue
import numpy as np
import json

from models.config import cfg

from util import text_processing
from util.feature_loader.feature_loader import ObjectsFeatureLoader

class BatchLoaderGqa:
    def __init__(self, imdb, data_params):
        self.imdb = imdb
        self.data_params = data_params

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'])
        self.T_encoder = data_params['T_encoder']
        self.answer_dict = text_processing.VocabDict(
            data_params['vocab_answer_file'])

        # positional encoding
        self.add_pos_enc = data_params.get('add_pos_enc', False)
        self.pos_enc_dim = data_params.get('pos_enc_dim', 0)
        assert self.pos_enc_dim % 4 == 0, \
            'positional encoding dim must be a multiply of 4'
        self.pos_enc_scale = data_params.get('pos_enc_scale', 1.)

        # feature loader
        objects_feature_dir = data_params['objects_feature_dir']
        self.objects_M = data_params.get('objects_max_num', 100)
        self.objects_loader = ObjectsFeatureLoader(objects_feature_dir,self.objects_M)

        # load one feature map to peek its size
        x, _ = self.objects_loader.load_feature(self.imdb[0]['imageId'])
        _, self.objects_D = x.shape

    def load_one_batch(self, sample_ids):
        actual_batch_size = len(sample_ids)
        input_seq_batch = np.zeros(
            (actual_batch_size, self.T_encoder), np.int32)
        seq_length_batch = np.zeros(actual_batch_size, np.int32)
        objects_feat_batch = np.zeros(
            (actual_batch_size, self.objects_M, self.objects_D),
            np.float32)
        objects_bbox_batch = np.zeros(
            (actual_batch_size, self.objects_M, 6), np.float32)
        objects_valid_batch = np.zeros(
            (actual_batch_size, self.objects_M), np.bool)

        # the required iterations for each question
        step_batch = np.ones(
            (actual_batch_size, cfg.ITER_NUM), np.int8)

        qid_list = [None]*actual_batch_size
        qstr_list = [None]*actual_batch_size
        imageid_list = [None]*actual_batch_size
        answer_label_batch = np.zeros(actual_batch_size, np.int32)
        for n in range(len(sample_ids)):
            iminfo = self.imdb[sample_ids[n]]
            question_str = iminfo['question']
            question_tokens = text_processing.tokenize(question_str)
            if len(question_tokens) > self.T_encoder:
                print('data reader: truncating question:\n\t' + question_str)
                question_tokens = question_tokens[:self.T_encoder]
            question_inds = [
                self.vocab_dict.word2idx(w) for w in question_tokens]
            seq_length = len(question_inds)
            input_seq_batch[n, :seq_length] = question_inds
            seq_length_batch[n] = seq_length

            feature, normalized_bbox, valid = \
                self.objects_loader.load_feature_normalized_bbox(
                    iminfo['imageId'])

            objects_feat_batch[n:n+1] = feature
            objects_bbox_batch[n:n+1] = normalized_bbox
            objects_valid_batch[n:n+1] = valid

            step = iminfo['step']
            step = step if step != 0 else 1
            step = step if step <= cfg.ITER_NUM else cfg.ITER_NUM

            step_batch[n, step:] = 0
            qid_list[n] = iminfo['questionId']
            qstr_list[n] = question_str
            imageid_list[n] = iminfo['imageId']
            answer_idx = self.answer_dict.word2idx(iminfo['answer'])
            answer_label_batch[n] = answer_idx

        batch = dict(input_seq_batch=input_seq_batch,
                     seq_length_batch=seq_length_batch,
                     answer_label_batch=answer_label_batch,
                     qid_list=qid_list, qstr_list=qstr_list,
                     imageid_list=imageid_list)

        batch['step_batch'] = step_batch
        batch['objects_feat_batch'] = objects_feat_batch
        batch['objects_bbox_batch'] = objects_bbox_batch
        batch['objects_valid_batch'] = objects_valid_batch
        if self.add_pos_enc:
            # add bounding boxes to the object features
            # tile bbox to roughly match the norm of RCNN features
            objects_bbox_tile = self.pos_enc_scale * np.tile(
                objects_bbox_batch, (1, 1, self.pos_enc_dim//6))
            image_feat_batch = np.concatenate(
                (objects_feat_batch, objects_bbox_tile), axis=-1)
        else:
            image_feat_batch = objects_feat_batch
        image_valid_batch = objects_valid_batch
        batch['image_feat_batch'] = image_feat_batch
        batch['image_valid_batch'] = image_valid_batch
        return batch


class DataReader:
    def __init__(self, data_file, shuffle, max_num=0, prefetch_num=16,
                 **kwargs):
        print('Loading imdb from %s' % data_file)
        with open(data_file) as f:
            raw_data = json.load(f)
            qIds = sorted(raw_data)
            for qId, q in raw_data.items():
                q['questionId'] = qId
            imdb = [raw_data[qId] for qId in qIds]
        print('Done')
        self.imdb = imdb
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num
        self.data_params = kwargs
        if max_num > 0 and max_num < len(self.imdb):
            print('keeping %d samples out of %d' % (max_num, len(self.imdb)))
            self.imdb = self.imdb[:max_num]

        # Vqa data loader
        self.batch_loader = BatchLoaderGqa(self.imdb, self.data_params)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
        self.prefetch_thread = threading.Thread(
            target=_run_prefetch, args=(
                self.prefetch_queue, self.batch_loader, self.imdb,
                self.shuffle, self.data_params))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def batches(self, one_pass):
        while True:
            # Get a batch from the prefetching queue
            # if self.prefetch_queue.empty():
            #     print('data reader: waiting for IO...')
            batch, n_sample, n_epoch = self.prefetch_queue.get(block=True)
            if batch is None:
                if one_pass:
                    return
                else:
                    # get the next batch
                    batch, n_sample, n_epoch = self.prefetch_queue.get(
                        block=True)
            yield (batch, n_sample, n_epoch)


class BatchLoaderGqaDialog:
    def __init__(self, imdb, questions, sub_masks, data_params):
        self.imdb = imdb
        self.questions = questions
        # self.sub_masks = sub_masks
        self.data_params = data_params

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'])
        self.T_encoder = data_params['T_encoder']
        self.answer_dict = text_processing.VocabDict(
            data_params['vocab_answer_file'])

        # positional encoding
        self.add_pos_enc = data_params.get('add_pos_enc', False)
        self.pos_enc_dim = data_params.get('pos_enc_dim', 0)
        assert self.pos_enc_dim % 4 == 0, \
            'positional encoding dim must be a multiply of 4'
        self.pos_enc_scale = data_params.get('pos_enc_scale', 1.)

        objects_feature_dir = data_params['objects_feature_dir']
        self.objects_M = data_params.get('objects_max_num', 100)
        self.objects_loader = ObjectsFeatureLoader(objects_feature_dir,self.objects_M)
        # load one feature map to peek its size
        x, _ = self.objects_loader.load_feature(self.questions[list(self.questions.keys())[0]]['imageId'])
        _, self.objects_D = x.shape


    def load_one_batch(self, sample_ids):

        sample_size = len(sample_ids)
        questions_nums = [len(self.imdb[sample_ids[n]]) for n in range(len(sample_ids))]
        actual_batch_size = sum(questions_nums)

        input_seq_batch = np.zeros(
            (actual_batch_size, self.T_encoder), np.int32)
        seq_length_batch = np.zeros(actual_batch_size, np.int32)
        objects_feat_batch = np.zeros(
            (actual_batch_size, self.objects_M, self.objects_D),
            np.float32)
        objects_bbox_batch = np.zeros(
            (actual_batch_size, self.objects_M, 6), np.float32)
        objects_valid_batch = np.zeros(
            (actual_batch_size, self.objects_M), np.bool)

        # the required iterations for each question
        step_batch = np.ones(
            (actual_batch_size, cfg.ITER_NUM), np.int8)
        # whether a question is a compostional (original) question
        ori_q_batch = np.ones((actual_batch_size), np.int8)
        # the indices of original question for a sub question
        sub_mask_batch = np.zeros([actual_batch_size, actual_batch_size])

        qid_list = [None]*actual_batch_size
        qid_list_sub = []
        qstr_list = [None]*actual_batch_size
        imageid_list = [None]*actual_batch_size
        answer_label_batch = np.zeros(actual_batch_size, np.int32)

        n = 0
        count = 0

        for i in range(len(sample_ids)):
            assert count == n
            indices_i = self.imdb[sample_ids[i]]

            sub_mask = np.zeros((len(indices_i),len(indices_i)))
            sub_mask[:-1,-1] = 1

            # sub_mask_batch[n:n + questions_nums[i], n:n + questions_nums[i]] = np.array(self.sub_masks[sample_ids[i]])
            sub_mask_batch[n:n + questions_nums[i], n:n + questions_nums[i]] = sub_mask

            count = count + questions_nums[i]

            for qid in indices_i:
                iminfo = self.questions[qid]

                question_str = iminfo['question']
                question_tokens = text_processing.tokenize(question_str)
                if len(question_tokens) > self.T_encoder:
                    print('data reader: truncating question:\n\t' + question_str)
                    question_tokens = question_tokens[:self.T_encoder]
                question_inds = [
                    self.vocab_dict.word2idx(w) for w in question_tokens]
                seq_length = len(question_inds)
                input_seq_batch[n, :seq_length] = question_inds
                seq_length_batch[n] = seq_length

                feature, normalized_bbox, valid = \
                    self.objects_loader.load_feature_normalized_bbox(
                        iminfo['imageId'])

                objects_feat_batch[n:n+1] = feature
                objects_bbox_batch[n:n+1] = normalized_bbox
                objects_valid_batch[n:n+1] = valid

                step = iminfo['step']
                step = step if step != 0 else 1
                step = step if step <= cfg.ITER_NUM else cfg.ITER_NUM
                step_batch[n, step:] = 0

                # sub question
                if '_' in iminfo['questionId']:
                    ori_q_batch[n:n+1] = 0
                    qid_list_sub.append(iminfo['questionId'])

                qid_list[n] = iminfo['questionId']
                qstr_list[n] = question_str
                imageid_list[n] = iminfo['imageId']
                answer_idx = self.answer_dict.word2idx(iminfo['answer'])
                answer_label_batch[n] = answer_idx
                n = n + 1

        assert n == actual_batch_size

        batch = dict(input_seq_batch=input_seq_batch,
                     seq_length_batch=seq_length_batch,
                     answer_label_batch=answer_label_batch,
                     qid_list=qid_list, qstr_list=qstr_list,
                     imageid_list=imageid_list)

        batch['qid_list_sub'] = qid_list_sub
        batch['sub_mask_batch'] = sub_mask_batch
        batch['ori_q_batch'] = ori_q_batch
        batch['step_batch'] = step_batch
        batch['objects_feat_batch'] = objects_feat_batch
        batch['objects_bbox_batch'] = objects_bbox_batch
        batch['objects_valid_batch'] = objects_valid_batch

        if self.add_pos_enc:
            # add bounding boxes to the object features
            # tile bbox to roughly match the norm of RCNN features
            objects_bbox_tile = self.pos_enc_scale * np.tile(
                objects_bbox_batch, (1, 1, self.pos_enc_dim//6))
            image_feat_batch = np.concatenate(
                (objects_feat_batch, objects_bbox_tile), axis=-1)
        else:
            image_feat_batch = objects_feat_batch

        image_valid_batch = objects_valid_batch
        batch['image_feat_batch'] = image_feat_batch
        batch['image_valid_batch'] = image_valid_batch
        return batch


class DataReaderDialog:
    def __init__(self, data_file, shuffle, max_num=0, prefetch_num=16,
                 **kwargs):
        print('Loading imdb from %s' % data_file)

        with open(data_file) as f:
            raw_data = json.load(f)
            indices = raw_data['indices']
            questions = raw_data['questions']
            # sub_masks = raw_data['sub_masks']
            qIds = sorted(indices)
            for qId, q in questions.items():
                q['questionId'] = qId
            # sub_masks = [sub_masks[qId] for qId in qIds]
            imdb = [indices[qId] for qId in qIds]

        print('Done')
        self.imdb = imdb
        self.questions = questions
        # self.sub_masks = sub_masks
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num
        self.data_params = kwargs
        if max_num > 0 and max_num < len(self.imdb):
            print('keeping %d samples out of %d' % (max_num, len(self.imdb)))
            self.imdb = self.imdb[:max_num]

        # Vqa data loader
        # self.batch_loader = BatchLoaderGqaDialog(self.imdb,self.questions, self.sub_masks, self.data_params)
        self.batch_loader = BatchLoaderGqaDialog(self.imdb,self.questions, None, self.data_params)
        

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
        self.prefetch_thread = threading.Thread(
            target=_run_prefetch, args=(
                self.prefetch_queue, self.batch_loader, self.imdb,
                self.shuffle, self.data_params))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def batches(self, one_pass):
        while True:
            # Get a batch from the prefetching queue
            # if self.prefetch_queue.empty():
            #     print('data reader: waiting for IO...')
            batch, n_sample, n_epoch = self.prefetch_queue.get(block=True)
            if batch is None:
                if one_pass:
                    return
                else:
                    # get the next batch
                    batch, n_sample, n_epoch = self.prefetch_queue.get(
                        block=True)
            yield (batch, n_sample, n_epoch)


def _run_prefetch(prefetch_queue, batch_loader, imdb, shuffle, data_params):
    num_samples = len(imdb)
    batch_size = data_params['batch_size']

    n_sample = 0
    n_epoch = 0
    fetch_order = np.arange(num_samples)

    np.random.seed(cfg.seed)

    while True:
        # Shuffle the sample order for every epoch
        if n_sample == 0 and shuffle:
            fetch_order = np.random.permutation(num_samples)

        # Load batch from file
        # note that len(sample_ids) <= batch_size, not necessarily equal
        sample_ids = fetch_order[n_sample:n_sample+batch_size]
        batch = batch_loader.load_one_batch(sample_ids)
        prefetch_queue.put((batch, n_sample, n_epoch), block=True)

        n_sample += len(sample_ids)
        if n_sample >= num_samples:
            n_sample = 0
            n_epoch += 1
            # Put in a None batch to indicate an epoch is over
            prefetch_queue.put((None, n_sample, n_epoch), block=True)
