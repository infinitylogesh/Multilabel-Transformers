# Multilabel Transformers

A repository to finetune transformers on a multilabel classification task.

## Usage :

1. Update `MultiLabelProcessor` class in `utils_glue.py` with the path of the file containing all the labels ( one label per line - see data/labels.txt for reference ).

2. Execute `run_glue.py` with new task_name - `multilabel` (other argument values are same as in transformers library - example below )

```bash
!python3 run_glue.py --model_type bert --save_steps 100000 --model_name_or_path "bert-base-uncased" --task_name multilabel --do_train --do_eval --num_train_epochs 13 --do_lower_case --data_dir ./ --max_seq_length 128 --output_dir "./output"
```

Note : For now the scripts support BERT and DistilBERT models.

## Traing and test file format :

Traing and test dataset file can be a comma seperated file with first column containing the sentences and second column with the labels inside a (") quotation mark  (seperated by comma). Modify `MultiLabelProcessor` class in `utils_glue.py` if your dataset is in a different format.

| text  | labels  |
| ------------ | ------------ |
|  i love nlp | "label1,label2"   |

## Credits

The scripts were adapted from Huggingface's [Transformers](https://huggingface.co/transformers) library.