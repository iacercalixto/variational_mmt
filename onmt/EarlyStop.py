from __future__ import division
"""
This file implements early stopping using BLEU for OpenNMT-py.
"""
import time
import sys,os
import math
import tempfile
import numpy
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules
from onmt.Utils import BLEU_SCRIPT, METEOR_SCRIPT, MODEL_TYPES
from subprocess import Popen, PIPE, STDOUT
from glob import glob

class EarlyStop(object):
    def __init__(self, src, tgt, early_stop_criteria, start_early_stop_at, evaluate_every_nupdates, patience, multimodal_model_type=None, img_fname=None, gpuid=0):
        criteria=['perplexity', 'bleu', 'meteor']
        assert(early_stop_criteria in criteria), \
                'ERROR: Invalid parameter value: \'%s\'. Accepted values: %s.'%(
                        early_stop_criteria, str(criteria))

        mm_type=[None] + MODEL_TYPES
        assert(multimodal_model_type in mm_type), \
                'ERROR: Invalid parameter value: \'%s\'. Accepted values: %s.'%(
                        multimodal_model_type, str(mm_type))

        if not multimodal_model_type is None:
            assert(not img_fname is None), 'Must provide image features file name for multimodal_model_type: %s'%(multimodal_model_type)

        self.src = src
        self.tgt = tgt
        self.img_fname = img_fname
        self.early_stop_criteria = early_stop_criteria
        self.start_early_stop_at = start_early_stop_at
        self.evaluate_every_nupdates = evaluate_every_nupdates
        self.patience = patience
        self.multimodal_model_type = multimodal_model_type

        # set to True when early stopping
        self.signal_early_stopping = False

        # dictionary containing '# model updates':'metric score'
        self.results_bleu = {}
        self.results_meteor = {}
        #self.n_model_updates = 0
        self.batch_size = 1
        self.beam_size = 1
        try:
            self.gpuid = gpuid[0]
        except:
            self.gpuid = gpuid
        # /dev/null
        self.fnull = open(os.devnull, 'w')


    def _do_early_stop(self):
        """ Check whether it is time to early-stop or not.
            Use BLEU or METEOR scores according to `self.early_stop_criteria`. """
        results = self.results_bleu if self.early_stop_criteria=='bleu' else self.results_meteor

        if len(results)+1 < self.patience:
            return False

        # include only BLEU and METEOR scores for best model on validation set - sort by number of model updates (past-to-current)
        sorted_metrics = sorted(results.items(), key=lambda kv:kv[0])

        last_results = float([v for k,v in sorted_metrics][-1])
        all_but_last_results = [v for k,v in sorted_metrics][:-1]
        print("EarlyStop._do_early_stop() - all_but_last_results %s"%str(all_but_last_results))
        print("EarlyStop._do_early_stop() - last_results %s"%str(last_results))

        # if the current metric is not the best one insofar, additional checks are needed
        if max(all_but_last_results) >= last_results:
            # position where the metric score is maximum (from the beginning of training until now)
            max_position = numpy.array(all_but_last_results).argmax()

            # make sure the best saved model up-to-now are less than `patience` timesteps away from now
            if len(all_but_last_results)+1 - max_position > self.patience:
                self.signal_early_stopping = True
                print("EarlyStop.signal_early_stopping = True - all_but_last_results: %s, last_results: %s"%(str(all_but_last_results),str(last_results)))
                return True

        return False


    def add_run(self, curr_model_snapshot, n_model_updates):
        """
            curr_model_snapshot (str):  Full path to file containing the current model snapshot saved on disk.
            n_model_updates (int):      Number of model updates used to arrive at model `curr_model_snapshot`.

            Returns:                    whether the metric (BLEU,METEOR) computed for the current run is
                                        the best one computed insofar. This can be used to decide
                                        whether to overwrite the current best model for model selection.
        """
        # do nothing
        if self.early_stop_criteria == 'perplexity' or self.early_stop_criteria is None:
            return False

        # temporary file to contain the translations of the validation set
        temp_hyp = tempfile.NamedTemporaryFile(delete=False)
        temp_hyp_name = temp_hyp.name
        temp_hyp.close()
        # translate the validation set
        self.translate_(self.src, curr_model_snapshot, temp_hyp_name)

        print('temp_hyp_name: ', temp_hyp_name)

        # compute metrics
        model_names, model_bleus, translation_files = self.compute_bleus(temp_hyp_name, self.tgt, 'valid')
        self.results_bleu[ n_model_updates ] = float(model_bleus[0])
        model_names, model_meteors, translation_files = self.compute_meteors(temp_hyp_name, self.tgt, 'valid')
        self.results_meteor[ n_model_updates ] = float(model_meteors[0])

        print("EarlyStop.add_run() - BLEU: %.4f, METEOR: %.4f"%(float(model_bleus[0]), float(model_meteors[0])))
        os.unlink(temp_hyp_name)
        self._do_early_stop()

        # history of BLEU/METEOR scores
        results = self.results_bleu if self.early_stop_criteria=='bleu' else self.results_meteor

        # include only BLEU and METEOR scores for best model on validation set - sort by number of model updates (past-to-current)
        sorted_metrics = sorted(results.items(), key=lambda kv:kv[0])

        last_results = float([v for k,v in sorted_metrics][-1])
        all_but_last_results = [v for k,v in sorted_metrics][:-1]
        #all_but_last_results = [v for k,v in sorted_metrics][:-1][:self.patience]
        print("EarlyStop._do_early_stop() - all_but_last_results %s"%str(all_but_last_results))
        print("EarlyStop._do_early_stop() - last_results %s"%str(last_results))

        # is this the best computed BLEU/METEOR so far?
        if len(all_but_last_results) == 0:
            # there is only one result insofar
            curr_best = True
        else:
            if max(all_but_last_results) > last_results:
                curr_best = False
            else:
                curr_best = True

        return curr_best


    def compute_meteors(self, hypotheses_fname, references_fname, split='valid'):
        """ This function computes METEOR for all translations one by one without threading/queuing.
            It first converts subwords back into words before computing METEOR scores.
        """
        #assert(split in ['valid', 'test2016']), 'Must compute METEOR for either valid or test set test2016!'
        assert(split=='valid')

        # compute METEOR scores for each of the translations of the validation set
        curr_model_idx=0
        model_meteors, model_names, translation_files = [], [], []

        # post-process reference translations
        with tempfile.NamedTemporaryFile() as temp_ref:
            # call another python script to compute scores
            pcat  = Popen(['cat', references_fname], stdout=PIPE, stderr=self.fnull)
            # convert translations from subwords into words
            psubword = Popen(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=pcat.stdout, stdout=temp_ref)
            # write translations after BPE-to-word post-processing to temporary file
            psubword.communicate()

            for hypfile in glob(hypotheses_fname):
                # post-process hypothesis translations
                with tempfile.NamedTemporaryFile() as temp_hyp:
                    # call another python script to compute scores
                    pcat  = Popen(['cat', hypfile], stdout=PIPE, stderr=self.fnull)
                    # convert translations from subwords into words
                    psubword = Popen(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=pcat.stdout, stdout=temp_hyp)
                    # write translations after BPE-to-word post-processing to temporary file
                    psubword.communicate()

                    # compute METEOR using `meteor-1.5.jar`
                    pmeteor = Popen(['java', '-Xmx2G', '-jar', METEOR_SCRIPT, temp_hyp.name, temp_ref.name, "-l", "de", "-norm"], stdout=PIPE, stderr=self.fnull)
                    # extract METEOR scores from output string
                    # grep -w "Final score:[[:space:]]" | tr -s ' ' | cut -d' ' -f3
                    pgrep = Popen(['grep', '-w', "Final score:[[:space:]]"], stdin=pmeteor.stdout,  stdout=PIPE, stderr=self.fnull)
                    #print(pgrep.communicate()[0].strip().decode('utf8'))
                    #sys.exit(1)
                    ptr = Popen(['tr', '-s', ' '], stdin=pgrep.stdout,  stdout=PIPE, stderr=self.fnull)
                    pcut = Popen(['cut', '-d', ' ', '-f', '3'], stdin=ptr.stdout,  stdout=PIPE, stderr=self.fnull)
                    #pcut1 = Popen(['cut', '-d', ',', '-f1'], stdin=pbleu.stdout, stdout=PIPE, stderr=self.fnull)
                    #pcut2 = Popen(['cut', '-d', ' ', '-f3'], stdin=pcut1.stdout, stdout=PIPE, stderr=self.fnull)

                    # add it to array of BLEUs
                    final_meteor=pcut.communicate()[0].strip().decode('utf8')
                    final_meteor=float(final_meteor)*100
                    model_meteors.append(final_meteor)
                    model_names.append(hypfile.replace('.pt.translations-%s'%split, '.pt'))
                    translation_files.append(hypfile)

                    #print("Computed METEOR: %s"%final_meteor)

                    curr_model_idx += 1
                    #print("final bleu: %s"%str(final_bleu))
        assert(curr_model_idx == len(model_meteors)), 'Problem detected while computing METEORs.'
        return model_names, model_meteors, translation_files


    def compute_bleus(self, hypotheses_fname, references_fname, split='valid'):
        """ This function computes BLEU for all translations one by one without threading/queuing.
            It first converts subwords back into words before compute BLEU scores.
        """
        assert(split in ['valid', 'test2016']), 'Must compute BLEU for either valid or test set test2016!'
        # compute BLEU scores for each of the translations of the validation set
        curr_model_idx=0
        model_bleus, model_names, translation_files = [], [], []
        # post-process reference translations
        with tempfile.NamedTemporaryFile() as temp_ref:
            pcat  = Popen(['cat', references_fname], stdout=PIPE, stderr=self.fnull)
            # convert translations from subwords into words
            psubword = Popen(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=pcat.stdout, stdout=temp_ref)
            # write translations after BPE-to-word post-processing to temporary file
            psubword.communicate()

            for hypfile in glob(hypotheses_fname):
                # call another python script to compute validation set BLEU scores
                pcat  = Popen(['cat', hypfile], stdout=PIPE, stderr=self.fnull)
                # convert translations from subwords into words
                psubword = Popen(['sed', '-r', 's/(@@ )|(@@ ?$)//g'], stdin=pcat.stdout, stdout=PIPE)
                # compute BLEU using `multi-bleu.perl`
                pbleu = Popen([BLEU_SCRIPT, temp_ref.name], stdin=psubword.stdout, stdout=PIPE, stderr=self.fnull)
                #pbleu = Popen([BLEU_SCRIPT, references_fname], stdin=psubword.stdout, stdout=PIPE, stderr=self.fnull)
                # extract BLEU scores from output string
                pcut1 = Popen(['cut', '-d', ',', '-f1'], stdin=pbleu.stdout, stdout=PIPE, stderr=self.fnull)
                pcut2 = Popen(['cut', '-d', ' ', '-f3'], stdin=pcut1.stdout, stdout=PIPE, stderr=self.fnull)

                # add it to array of BLEUs
                final_bleu=pcut2.communicate()[0].strip().decode('utf8')
                model_bleus.append(final_bleu)
                model_names.append(hypfile.replace('.pt.translations-%s'%split, '.pt'))
                translation_files.append(hypfile)

                curr_model_idx += 1
                #print("final bleu: %s"%str(final_bleu))
        assert(curr_model_idx == len(model_bleus)), 'Problem detected while computing BLEUs.'
        return model_names, model_bleus, translation_files


    def translate_(self, source_fname, model_fname, hypfname_out):
        # call external python script to translate validation/test set
        if self.multimodal_model_type is None:
            script_fname='translate.py'
        elif self.multimodal_model_type in MODEL_TYPES:
            script_fname='translate_mm_vi.py'
        else:
            raise Exception("Multimodal model type not supported: %s"%self.multimodal_model_type)

        job_cmd = ['python', script_fname,
                   '-src', source_fname, #'-path_to_test_img_feats', img_feats_fname,
                   '-model', model_fname,
                   '-batch_size', str(self.batch_size),
                   '-beam_size', str(self.beam_size),
                   '-gpu', str(self.gpuid),
                   '-output', hypfname_out]

        if not self.multimodal_model_type is None:
            job_cmd += ['-path_to_test_img_feats', str(self.img_fname)]

        p = Popen(job_cmd,  stdout=PIPE)
        #p = Popen(job_cmd, stderr=self.fnull, stdout=PIPE)
        start = time.time()
        print("Started computing translations...")
        out, err = p.communicate()
        print("Finished: %d seconds elapsed."%(time.time()-start))
        #print("out: ", str(out))
        #print("err: ", str(err))
