from json import load
import pandas as pd
import dill
import os
import logging
import sklearn.linear_model as sklm

from continual.python.sdk.extensions.algorithms.base_custom_algorithm import (
    BaseCustomAlgorithm,
)

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer, TFDistilBertForSequenceClassification, create_optimizer, DistilBertTokenizer
from lime.lime_text import LimeTextExplainer
import random
import tensorflow as tf
from continual.python.sdk.logger.logger import logger as metadata_logger


"""
Template custom model class

ADD YOUR IMPLEMENTATION HERE.
"""


class DistilBert(BaseCustomAlgorithm):
    """
    My custom model
    """

    model_file_name: str = "custom_algo.pkl"

    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize any state

        Parameters
        ----------
        logger: Logger
            The output logger initialized with log
            files for this model
        kwargs: dict
            Key word arguments
        """

        self.logger = logger
        self.model_type = sklm.ElasticNet


    def _create_huggingface_dataset_from_split(self, X, y, X_val, y_val, params):
        """ Returns huggingface dataset
        Takes in train and validation dataframes for X and Y and converts to huggingface dataset
        """

        # create a huggingface Dataset from the pandas dataframe
        train, train["label"] = X, y
        train_dataset = Dataset.from_pandas(train)
        val, val["label"] = X_val, y_val
        valid, test = np.split(val, [int(.5*len(val))])
        valid_dataset = Dataset.from_pandas(valid)
        test_dataset = Dataset.from_pandas(test)

        # gather everyone if you want to have a single DatasetDict
        hf_dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'valid': valid_dataset})

        return hf_dataset
    
    def _create_tensorflow_dataset(self, tokenized_dict, data_collator):
        self.logger.info("Creating tf dataset")

        tf_train_set = tokenized_dict["train"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "label"],
            shuffle=True,
            batch_size=16,
            collate_fn=data_collator,
        )

        tf_validation_set = tokenized_dict["test"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "label"],
            shuffle=False,
            batch_size=16,
            collate_fn=data_collator,
        )
        return tf_train_set, tf_validation_set
    
    def _tokenize_dataset(self, dataset):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        return dataset.map(preprocess_function, batched=True), tokenizer
    
    def _load_pretrained_model(self):
        from continual.python.sdk.logger.logger import logger as metadata_logger
        from transformers import TFDistilBertForSequenceClassification
        import os

        model_version = 'projects/topic_classification1/models/product_classification_example/versions/ccd6al4doqdt0tst8arg'
        artifact, model_path = metadata_logger.get_model_artifact(model_version=model_version, download=True, download_dir="/home/tylerkohn/continual/brendan_test")

        print(f"model_path: {model_path}")

        full_model_path = os.path.join(model_path, "artifacts/tmpd1mpjpzl/AutogluonModels/ag-20220908_215906/models/distilbert/custom_trainer_dir/transformer_model")
        print(f"full_model_path: {full_model_path}")
        loaded_model = TFDistilBertForSequenceClassification.from_pretrained(full_model_path)
        print("Model successfully downloaded")
        return loaded_model
        

    def _run_lime(self, X):
        #Run LIME analysis
        def predictor(X):
            self.logger.info("creating lime tokenizer")
            lime_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

            lime_encodings = lime_tokenizer(X,truncation=True, padding=True)
            lime_dataset = tf.data.Dataset.from_tensor_slices((dict(lime_encodings)))
            preds_lime = self.model.predict(lime_dataset).logits
            self.logger.info("Finished preds_lime")
            self.logger.info(np.array(preds_lime))
            return(np.array(preds_lime))
        
        self.logger.info("Creating the input sentence:")
        self.logger.info(X.shape)
        
        self.logger.info(X['text'])
        
        input_sentence = X['text'].iloc[random.randint(0, len(X))]
        self.logger.info("Input sentence created")
        self.logger.info(input_sentence)
        
        #tokenize the text
        class_names = np.array(['Credit reporting', 'Debt collection', 'Student loan', 'Mortgage','Credit card', 'Checking or', 'Money transfer,', 'Vehicle loan', 'Payday loan'])
        explainer = LimeTextExplainer(class_names=class_names)
        self.logger.info("Running explain_instance")
        exp = explainer.explain_instance(input_sentence, predictor, num_features=6, num_samples=1000, top_labels=2)
        self.logger.info("Creating pyplot")
        fig = exp.as_pyplot_figure(label=exp.available_labels()[0])
        fig.suptitle("'" +input_sentence[0:75]+"...'", fontsize=10, y=0.95)
        filename = "lime_sentence_sample.png"
        fig.savefig(filename)
        metadata_logger.log_artifact(filename,type="plot")
        self.logger.info("LIME artifact has been logged")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        params: dict,
    ):
        """
        Construct and train model defined by config on data.
        Store any model state in self

        Parameters
        ----------
        X: DataFrame
            The training
        y: DataFrame
            The training outputs
        X_val: DataFrame
            validation features
        y_val: DataFrame
            validation outputs
        params: dict
            hyperparameters to the model
        """
        if self.model_type:
            # Convert dataframe to huggingface dataset
            train_test_valid_dataset = self._create_huggingface_dataset_from_split(X, y, X_val, y_val, params)
            
            # Create tokenizer and tokenize dataset
            tokenized_dict, tokenizer = self._tokenize_dataset(train_test_valid_dataset)

            # Batch and pad dataset
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

            # Create Tensorflow Datasets
            tf_train_set, tf_validation_set = self._create_tensorflow_dataset(tokenized_dict, data_collator)

            batch_size = 16
            num_epochs = 1
            batches_per_epoch = len(tokenized_dict["train"]) // batch_size
            total_train_steps = int(batches_per_epoch * num_epochs)
            optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

            self.logger.info("Loading model from_pretrained")
            #model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=y.nunique())
            model = self._load_pretrained_model()
            self.logger.info("Compiling model")
            model.compile(optimizer=optimizer)

            self.logger.info("Fitting model")
            
            model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=1)
            self.model = model
            self.logger.info("Model has been fit")
            self.logger.info(self.model)

    def predict(self, X: pd.DataFrame, params: dict) -> pd.Series:
        """
        Run trained model on data. Utilize
        any model state in self.

        Note: Continual will ensure that fit()
        has been called before predict()

        Parameters
        ----------
        X : DataFrame
            The features from which to get predictions
        params: dict
            hyperparameters to the model

        Returns
        -------
        pd.Series
            The class predictions for the input data
        """
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        #tokenize the text
        encodings = tokenizer(X['text'].tolist(),
                      truncation=True, 
                      padding=True)

        #transform to tf.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))

        #predict
        self.logger.info("Running predict")
        results_df = pd.DataFrame(self.model.predict(dataset).logits, columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12']).to_numpy()
        self.logger.info("Finished predict.")
        
        #run lime analysis on a sample
        try:
            self.logger.info("Running LIME analysis")
            self._run_lime(X)
        except:
            self.logger.info("LIME analysis failed to generate. Skipping...")    
        
        return results_df

    def save(self, save_dir) -> str:
        """
        Serialize current instance state to disk in given
        directory (directory is already created)

        Parameters
        ----------
        save_dir : str
            Save path for serialized model file
        """

        self.model.save_pretrained(os.path.join(save_dir, "transformer_model"))
        self.model = None

    def load(self, load_dir: str):
        """
        Deserialize current instance state from given directory

        Parameters
        ----------
        load_dir: str
            The directory where the serialized files for this
            model are stored
        """
        self.model = self._load_pretrained_model()
        

    def default_parameters(self):
        """
        Returns
        -------
        dict
            The default hyperparameters that will be
            passed to the fit function
        """

        return {"random_state": 0}
