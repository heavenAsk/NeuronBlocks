{
  "license": "Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.",
  "tool_version": "1.1.0",
  "model_description": "This model is used for model compression",
  "language": "Chinese",
  "inputs": {
    "use_cache": false,
    "dataset_type": "classification",
    "data_paths": {
      "train_data_path": "./dataset/chinese_nli/cnli_train_demo.txt",
      "valid_data_path": "./dataset/chinese_nli/cnli_dev_demo.txt",
      "test_data_path": "./dataset/chinese_nli/cnli_test_demo.txt",
      "predict_data_path": "./dataset/chinese_nli/cnli_test_demo.txt",
      "pre_trained_emb": "./dataset/sogou_embed/sgns.sogou.word"
    },
    "add_start_end_for_seq": true,
    "file_header": {
      "premise_text": 0,
      "hypothesis_text": 1,
      "label": 2
    },
    "predict_file_header": {
      "premise_text": 0,
      "hypothesis_text": 1,
      "label": 2
    },
    "model_inputs": {
      "premise": ["premise_text"],
      "hypothesis": ["hypothesis_text"]
    },
    "target": ["label"]
  },
  "outputs":{
    "save_base_dir": "./models/chinese_nli/",
    "model_name": "model.nb",
    "train_log_name": "train.log",
    "test_log_name": "test.log",
    "predict_log_name": "predict.log",
    "predict_fields": ["prediction"],
    "predict_output_name": "predict.txt",
    "cache_dir": ".cache.chinese_nli/"
  },
  "training_params": {
    "vocabulary": {
      "min_word_frequency": 1
    },
    "optimizer": {
      "name": "SGD",
      "params": {
        "lr": 0.2,
        "momentum": 0.9,
        "nesterov": true
      }
    },
    "lr_decay": 0.95,
    "minimum_lr": 0.005,
    "epoch_start_lr_decay": 1,
    "use_gpu": true,
    "batch_size": 128,
    "batch_num_to_show_results": 100,
    "max_epoch": 5,
    "valid_times_per_epoch": 1,
    "max_lengths": {
        "premise": 32,
        "hypothesis": 32
    }
  },
  "architecture":[
    {
        "layer": "Embedding",
        "conf": {
          "word": {
            "cols": ["premise_text", "hypothesis_text"],
            "dim": 300
          }
        }
    },
    {
        "layer_id": "premise_dropout",
        "layer": "Dropout",
        "conf": {
            "dropout": 0
        },
        "inputs": ["premise"]
    },
    {
        "layer_id": "hypothesis_dropout",
        "layer": "Dropout",
        "conf": {
            "dropout": 0
        },
        "inputs": ["hypothesis"]
    },
    {
        "layer_id": "premise_bigru",
        "layer": "BiGRU",
        "conf": {
            "hidden_dim": 128,
            "dropout": 0.3,
            "num_layers": 2
        },
        "inputs": ["premise_dropout"]
    },
    {
        "layer_id": "hypothesis_bigru",
        "layer": "premise_bigru",
        "inputs": ["hypothesis_dropout"]
    },
    {
        "layer_id": "premise_attn",
        "layer": "BiAttFlow",
        "conf": {
        },
        "inputs": ["premise_bigru","hypothesis_bigru"]
    },
    {
        "layer_id": "hypothesis_attn",
        "layer": "BiAttFlow",
        "conf": {
        },
        "inputs": ["hypothesis_bigru", "premise_bigru"]
    },
    {
        "layer_id": "premise_bigru_final",
        "layer": "BiGRU",
        "conf": {
            "hidden_dim": 128,
            "num_layers": 1
        },
        "inputs": ["premise_attn"]
    },
    {
        "layer_id": "hypothesis_bigru_final",
        "layer": "BiGRU",
        "conf": {
            "hidden_dim": 128,
            "num_layers": 1
        },
        "inputs": ["hypothesis_attn"]
    },
    {
        "layer_id": "premise_pooling",
        "layer": "Pooling",
        "conf": {
          "pool_axis": 1,
          "pool_type": "max"
        },
        "inputs": ["premise_bigru_final"]
    },
    {
        "layer_id": "hypothesis_pooling",
        "layer": "Pooling",
        "conf": {
          "pool_axis": 1,
          "pool_type": "max"
        },
        "inputs": ["hypothesis_bigru_final"]
    },
    {
        "layer_id": "comb",
        "layer": "Combination",
        "conf": {
            "operations": ["origin", "difference", "dot_multiply"]
        },
        "inputs": ["premise_pooling", "hypothesis_pooling"]
    },
    {
        "output_layer_flag": true,
        "layer_id": "output",
        "layer": "Linear",
        "conf": {
          "hidden_dim": [128, 3],
          "activation": "PReLU",
          "batch_norm": true,
          "last_hidden_activation": false
        },
        "inputs": ["comb"]
    }
  ],
  "loss": {
    "losses": [
      {
        "type": "CrossEntropyLoss",
        "conf": {
          "size_average": true
        },
        "inputs": ["output","label"]
      }
    ]
  },
  "metrics": ["accuracy"]
}




