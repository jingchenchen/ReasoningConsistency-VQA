# ReasoningConsistency-VQA

* C. Jing, Y. Jia, Y. Wu, X. Liu, Q. Wu, *Maintaining Reasoning Consistency in Compositional Visual Question Answering*. in CVPR 2022 ([PDF](https://jingchenchen.github.io/files/papers/2022/CVPR_DLR.pdf))

## GQA-Sub Dataset 
We build a GQA-Sub dataset to enable the quantitative evaluation of reasoning consistency in compositional VQA. The GQA-Sub dataset is constructed based on the GQA dataset, a large-scale dataset for real-world visual reasoning and compositional question answering. We only generate sub-questions for questions of the train split and the validation split of GQA because the ground-truth scene graphs of the two splits are available. Thus our GQA-Sub dataset contains a train-sub split and a validation-sub split. The two splits can be found in the folder "questions". 

## Dialog-like Reasoning Method
We propose a dialog-like reasoning method that integrates the reasoning processes for sub-questions into the reasoning process for a compositional question to maintain the reasoning consistency in compositional VQA. The folder "DLR" contains the source code of the proposed method. 


### Prerequisites
- NVIDIA Driver & CUDA & cuDNN
- Python 3.6
- Pytorch 1.9.1.post3
- numpy 1.19.5
- h5py 3.1.0
- PyYAML 6.1
- file_utils 0.0.1

### Train and evaluate 

#### Step 1: download data

Please download all the question files from [here](https://www.dropbox.com/s/vdd4uviuk161ov7/questions.zip) and the visual features from [here](https://convaisharables.blob.core.windows.net/meta-module-network/gqa_visual_features.zip).

#### Step 2: training 

``` sh
cd dir/
python exp/main.py exp_id 001 dialog True TRAIN.SPLIT_VQA train_dialog_balanced
``` 

#### Step 3: evaluation

``` sh
python exp/main.py train False TEST.EVAL_ID 001 TEST.EPOCH 10 TEST.DUMP_PRED True TEST.SPLIT_VQA val_sub_balanced
python exp/main.py train False TEST.EVAL_ID 001 TEST.EPOCH 10 TEST.DUMP_PRED True TEST.SPLIT_VQA val_balanced 
python util/compute_consistency.py --exp_name 001_DLR --epoch 10
``` 


## Citation

If you find the dataset or code useful, please consider citing our paper:

```bibtex
@inproceedings{jing2022maintaining,
  title={Maintaining Reasoning Consistency in Compositional Visual Question Answering},
  author={Jing, Chenchen and Jia, Yunde and Wu, Yuwei and Liu, Xinyu and Wu, Qi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## Acknowledgment

The implementation of the dialog-like reasoning is partly based on the following codebases. We gratefully thank the authors for their wonderful works: 
[LCGN](https://github.com/ronghanghu/lcgn),
[MMN](https://github.com/wenhuchen/Meta-Module-Network),
[Logic-guided QA](https://github.com/AkariAsai/logic_guided_qa),

