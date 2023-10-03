#!/usr/bin/env python
# coding: utf-8

# #### 0. Config

# In[ ]:


import os
import sys
import math
import logging
import tempfile
import warnings
from pprint import pprint
from datetime import date, timedelta
from typing import List, Dict, Text, TypeVar, Union, Optional, Any, Iterable
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pyspark.ml.feature import Bucketizer
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from nba.utils.common_util import qa_uniqueness, lowercase_all, add_hash_key, get_latest_mlflow_model_version, get_artifact_uri_by_model_name, customize_config_dict, winsorize
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep
import tensorflow_recommenders as tfrs
from nba.utils.common_util import winsorize



# In[ ]:


from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as f


assembler = VectorAssembler(inputCols=features_to_scale, outputCol="features")
df_assembled = assembler.transform(data_sampled)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT

# Define a UDF to extract values from the scaled features vector and replace the original columns
def extract(i):
    def extract_feature_udf(v):
        try:
            return float(v[i])
        except ValueError:
            return None
    return udf(extract_feature_udf, DoubleType())

# Replace original columns with the scaled ones
for i, col in enumerate(features_to_scale):
    df_scaled = df_scaled.withColumn(col, extract(i)("scaled_features"))


data_sampled_normalized = df_scaled.drop("features", "scaled_features")



# In[ ]:


from nba.core.training.feature_transformers import PCATransfomer
pca_obj = PCATransfomer()


# In[ ]:


pca_model_ppm = pca_obj.fit_pipeline(data_sampled_normalized, ppm, 8)
data_tf_fs_staging, pca_model_ppm, ppm_features = pca_obj.perform_pca_transformation(pca_model_ppm, data_sampled_normalized, ppm, 'ppm_features_pca')        


# In[ ]:


pca_model_product_bookings = pca_obj.fit_pipeline(data_tf_fs_staging, product_bookings, 8)
data_tf_fs_staging, pca_model_product_bookings, product_bookings_features = pca_obj.perform_pca_transformation(pca_model_product_bookings, data_tf_fs_staging , product_bookings, 'product_bookings_pca')    


# In[ ]:


pca_model_holiday = pca_obj.fit_pipeline(data_tf_fs_staging,holiday, 2)
data_tf_fs_staging, pca_model_holiday, holiday_features = pca_obj.perform_pca_transformation(pca_model_holiday, data_tf_fs_staging , holiday, 'holiday_pca')    


# In[ ]:


pca_model_site = pca_obj.fit_pipeline(data_tf_fs_staging, site, 8)
data_tf_fs_staging, pca_model_site, site_features = pca_obj.perform_pca_transformation(pca_model_site, data_tf_fs_staging , site, 'site_pca')    


# In[ ]:


feature_catalog = {}
feature_catalog['user_cat_features'] = user_cat_selected_features_raw
feature_catalog['user_numeric_features'] = [c for c in stand_alone_features + \
                                            site_features + \
                                            holiday_features + \
                                            ppm_features + \
                                            product_bookings_features if c not in user_cat_selected_features_raw]
feature_catalog['utm_cat_features'] = utm_cat_cols
feature_catalog['utm_numeric_features'] = utm_numeric_cols
feature_catalog['other_columns'] = other_columns
feature_catalog['label'] = target_label_retrieve


# In[ ]:


data_tf_fs_staging = data_tf_fs_staging.withColumn("hash_key", 
                                                   data_tf_fs_staging["hash_key"].alias("hash_key", metadata=feature_catalog)) 
# To access metadata
for field in data_tf_fs_staging.select('hash_key').schema.fields:
    print(field.name, field.metadata)

data_final = data_tf_fs_staging.filter(f"""{feature_catalog['label']} > 0""")


# ####train-test split

# In[ ]:


#FOR SPARK
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StringType, DoubleType, FloatType

def convert_vector_to_float32(vector):
    return Vectors.dense([float(x) for x in vector])

vector_to_float32_udf = udf(convert_vector_to_float32, VectorUDT())


# Convert to float32
for col_name, col_type in data_final.dtypes:
    if col_type in ["vector"]:
        data_final = data_final.withColumn(col_name, vector_to_float32_udf(f.col(col_name)))
    if col_type in ["float", "int"]:
        data_final = data_final.withColumn(col_name, col(col_name).cast(FloatType()))


#SPLIT BASED ON LVID for train test
unique_lvid = data_final.select('looker_visitor_id').distinct()
unique_lvid_train, unique_lvid_test = unique_lvid.randomSplit([0.9, 0.1], seed=12345)

df_train = data_final.join(unique_lvid_train, on=['looker_visitor_id']).withColumn('cutoff_date', f.date_format('cutoff_date', 'yyyy-MM-dd'))
df_test = data_final.join(unique_lvid_test, on=['looker_visitor_id']).withColumn('cutoff_date', f.date_format('cutoff_date', 'yyyy-MM-dd'))

num_samples = df_train.count()
batch_size = 50
steps_per_epoch = int(num_samples // batch_size)

# for t in [df_train, df_test]:
#     df_stats = t.select(
#         _mean(col(target_label_rank)).alias('mean'),
#         _stddev(col(target_label_rank)).alias('std')
#     ).collect()

#     mean = df_stats[0]['mean']
#     std = df_stats[0]['std']
#     print(f'label {t}. mean: {mean}, std: {std}')


# ####inputs mapping

# In[ ]:


from functools import partial
def map_function_petastorm(row, 
                            num_features, 
                            cat_features,
                            other_columns,
                            label_column, 
                            mode):
    inputs = {}
   
    # Process user categorical input names
    for input_name in cat_features:
        inputs[input_name] = tf.reshape(getattr(row, input_name), [-1, 1])

    # Process user numerical input names
    for input_name in num_features:
        inputs[input_name] = tf.reshape(tf.cast(getattr(row, input_name), tf.float32), [-1, 1])
    
    for input_name in other_columns:
        value = getattr(row, input_name)
        if tf.is_tensor(value) and value.dtype == tf.string:
            # If it's a string tensor, just reshape it.
            inputs[input_name] = tf.reshape(value, [-1, 1])
        else:
            # Otherwise, cast to float and then reshape.
            inputs[input_name] = tf.reshape(tf.cast(value, tf.float32), [-1, 1])
            
    if mode == "predict":
        return inputs
    else:
        label = tf.reshape(tf.cast(getattr(row, label_column), tf.float32), [-1, 1])
        return (inputs, label)

def create_custom_map_function_petastorm(target_label, num_features, cat_features, other_columns):
    
    custom_map_function = {
        "train": partial(map_function_petastorm, 
                            num_features=num_features, 
                            cat_features=cat_features,    
                            other_columns = other_columns,                         
                            label_column=target_label, 
                            mode="train"),
        "test": partial(map_function_petastorm, 
                            num_features=num_features, 
                            cat_features=cat_features,
                            other_columns = other_columns,
                            label_column=target_label, 
                            mode="test"),
        "val": partial(map_function_petastorm, 
                            num_features=num_features, 
                            cat_features=cat_features,
                            other_columns = other_columns,
                            label_column=target_label, 
                            mode="val"),
        "predict": partial(map_function_petastorm, 
                            num_features=num_features, 
                            cat_features=cat_features,
                            other_columns = other_columns,
                            label_column=target_label, 
                            mode="predict")
    }

    return custom_map_function


def map_function_programmatic(row, column_names, input_names, label_column, mode):
    inputs = {input_name: row[column_names.index(input_name)] for input_name in input_names}
    
    if mode == "predict":
        return inputs
    else:
        label = row[column_names.index(label_column)]
        return (inputs, label)

def create_custom_map_function(model, dtf, target_label):
    column_names = dtf.columns
    input_names = [input_name for input_name in model.input_names if input_name != target_label]

    custom_map_function = {
        "train": partial(map_function_programmatic, column_names=column_names, input_names=input_names, label_column=target_label, mode="train"),
        "test": partial(map_function_programmatic, column_names=column_names, input_names=input_names, label_column=target_label, mode="test"),
        "val": partial(map_function_programmatic, column_names=column_names, input_names=input_names, label_column=target_label, mode="val"),
        "predict": partial(map_function_programmatic, column_names=column_names, input_names=input_names, label_column=target_label, mode="predict")
    }
    
    return custom_map_function


# ####encoder

# In[ ]:


from tabtransformertf.utils.preprocessing import df_to_dataset

def df_to_dataset_custom(
    dataframe: pd.DataFrame,
    target: str = None,
    shuffle: bool = True,
    batch_size: int = 512,
):
    df = dataframe.copy()
    if target:
        labels = df.pop(target)
        dataset = {}
        for key, value in df.items():
            dataset[key] = value[:, tf.newaxis]

        dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
    else:
        dataset = {}
        for key, value in df.items():
            dataset[key] = value[:, tf.newaxis]

        dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(batch_size)
    return dataset


# In[ ]:


#creating a small pandas df that contains all unique values of the cateegorical variable (for the encoders to create lookup tables)
target_label=target_label_rank

distinct_utms = df_train.select(feature_catalog['utm_cat_features']+feature_catalog['utm_numeric_features']).distinct()
utms_dataset_pd  = distinct_utms.toPandas()


#CREATE A PANDAS DF THAT CONTAINS ALL UNIQUE CATEGORICAL VALUES FOR EACH CATEGORICAL COLUMN
unique_values_dict = {}
for column in feature_catalog['user_cat_features']+feature_catalog['utm_cat_features']:
    unique_values = [row[column] for row in df_train.select(column).distinct().collect()]
    unique_values_dict[column] = unique_values

# Find the maximum length of the lists in the dictionary to determine the number of rows in the new DataFrame
max_len = max(len(v) for v in unique_values_dict.values())

# Pad the shorter lists with None to make all lists equal in length
for k, v in unique_values_dict.items():
    unique_values_dict[k] = v + [None] * (max_len - len(v))

# Create a Pandas DataFrame from the dictionary
unique_values_df = pd.DataFrame.from_dict(unique_values_dict)

# Pass the unique_values_df to the CEmbedding class
feature_names = list(unique_values_dict.keys())


dataset_candidates = df_to_dataset_custom(utms_dataset_pd)




# In[ ]:


class FTTransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        numerical_features: list,
        categorical_data,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        explainable=False,
    ):
        """FTTransformer Encoder
        Args:
            categorical_features (list): names of categorical features
            numerical_features (list): names of numeric features
            categorical_lookup (dict): dictionary with categorical feature names as keys and adapted StringLookup layers as values
            out_dim (int): model output dimensions
            out_activation (str): model output activation
            embedding_dim (int, optional): embedding dimensions. Defaults to 32.
            depth (int, optional): number of transformer blocks. Defaults to 4.
            heads (int, optional): number of attention heads. Defaults to 8.
            attn_dropout (float, optional): dropout rate in transformer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in mlps. Defaults to 0.1.
            mlp_hidden_factors (list[int], optional): numbers by which we divide dimensionality. Defaults to [2, 4].
            numerical_embeddings (dict, optional): dictionary with numerical feature names as keys and adapted numerical embedding layers as values. Defaults to None.
            numerical_embedding_type (str, optional): name of the numerical embedding procedure. Defaults to linear.
            use_column_embedding (bool, optional): flag to use fixed column positional embeddings. Defaults to True.
            explainable (bool, optional): flag to output importances inferred from attention weights. Defaults to False.
        """

        super(FTTransformerEncoder, self).__init__()
        self.numerical = numerical_features
        self.categorical = categorical_features
        self.categorical_data = categorical_data
        self.embedding_dim = embedding_dim
        self.explainable = explainable
        self.depth = depth
        self.heads = heads
            
        # Two main embedding modules
        if len(self.numerical) > 0:
            self.numerical_embeddings = NEmbedding(
                feature_names=self.numerical,                
                emb_dim=embedding_dim
            )
        if len(self.categorical) > 0:
            self.categorical_embeddings = CEmbedding(
                feature_names=self.categorical,
                X=self.categorical_data,
                emb_dim =embedding_dim
            )

        # Transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embedding_dim,
                    heads,
                    embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=self.explainable,
                    post_norm=False,  # FT-Transformer uses pre-norm
                )
            )
        self.flatten_transformer_output = Flatten()

        # CLS token
        w_init = tf.random_normal_initializer()
        self.cls_weights = tf.Variable(
            initial_value=w_init(shape=(1, embedding_dim), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs): 
        # print('intput encoder: ', inputs['utm_placement_type'])
        # check.append(inputs)
        try:
            cls_tokens = tf.repeat(self.cls_weights, repeats=tf.shape(inputs[self.numerical[0]])[0], axis=0)
        except:
            inputs = inputs[0]
            cls_tokens = tf.repeat(self.cls_weights, repeats=tf.shape(inputs[self.numerical[0]])[0], axis=0)
        cls_tokens = tf.expand_dims(cls_tokens, axis=1)
        transformer_inputs = [cls_tokens]
    
        # If categorical features, add to list
        if len(self.categorical) > 0:
            cat_input = []
            for c in self.categorical:
                cat_input.append(inputs[c])
            
            cat_input = tf.stack(cat_input, axis=1)[:, :, 0]
            cat_embs = self.categorical_embeddings(cat_input)
            transformer_inputs += [cat_embs]
        
        # If numerical features, add to list
        if len(self.numerical) > 0:
            num_input = []
            for n in self.numerical:
                num_input.append(inputs[n])
            num_input = tf.stack(num_input, axis=1)[:, :, 0]
            num_embs = self.numerical_embeddings(num_input)
            transformer_inputs += [num_embs]
        
        # Prepare for Transformer
        transformer_inputs = tf.concat(transformer_inputs, axis=1)
        importances = []
        
        # Pass through Transformer blocks
        for transformer in self.transformers:
            if self.explainable:
                transformer_inputs, att_weights = transformer(transformer_inputs)
                importances.append(tf.reduce_sum(att_weights[:, :, 0, :], axis=1))
            else:
                transformer_inputs = transformer(transformer_inputs)

        if self.explainable:
            # Sum across the layers
            importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
                self.depth * self.heads
            )
            return transformer_inputs, importances
        else:
            return transformer_inputs


# In[ ]:


class NEmbedding(tf.keras.Model):
    def __init__(
        self,
        feature_names: list,       
        emb_dim: int = 32,
    ):
        super(NEmbedding, self).__init__()

        self.num_features = len(feature_names)
        self.features = feature_names
        self.emb_dim = emb_dim

        # Initialise linear layer
        w_init = tf.random_normal_initializer()
        self.linear_w = tf.Variable(
            initial_value=w_init(
                shape=(self.num_features, 1, self.emb_dim), dtype='float32' # features, n_bins, emb_dim
            ), trainable=True)
        self.linear_b = tf.Variable(
            w_init(
                shape=(self.num_features, 1), dtype='float32' # features, n_bins, emb_dim
            ), trainable=True)
    
    
    def embed_column(self, f, data):
        emb = self.linear_layers[f](self.embedding_layers[f](data))
        return emb
   
    def call(self, x):
        embs = tf.einsum('f n e, b f -> bfe', self.linear_w, x)
        embs = tf.nn.relu(embs + self.linear_b)
        return embs
    
    
class CEmbedding(tf.keras.Model):
    def __init__(
        self,
        feature_names: list,
        X: pd.DataFrame,  # X is a Pandas DataFrame
        emb_dim: int = 32,
    ):
        super(CEmbedding, self).__init__()
        self.features = feature_names
        self.emb_dim = emb_dim
        self.category_prep_layers = {}
        self.emb_layers = {}
        
        for c in self.features:
            # Use the unique values from the DataFrame column to initialize the StringLookup layers
            lookup = tf.keras.layers.StringLookup(vocabulary=X[c].dropna().tolist())
            emb = tf.keras.layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=self.emb_dim)
            self.category_prep_layers[c] = lookup
            self.emb_layers[c] = emb
    
    def embed_column(self, f, data):
        return self.emb_layers[f](self.category_prep_layers[f](data))
    
    def call(self, x):
        emb_columns = []
        for i, f in enumerate(self.features):
            emb_columns.append(self.embed_column(f, x[:, i]))
        embs = tf.stack(emb_columns, axis=1)
        return embs


# In[ ]:


from sklearn.metrics import roc_auc_score
from tabtransformertf.models.tabtransformer import TransformerBlock
from tensorflow.keras.layers import (
    Dense,
    Flatten,
)
global verbose
verbose=True


out_dim=4
out_activation="sigmoid"

#user encoder
user_linear_encoder = FTTransformerEncoder(
    numerical_features = feature_catalog['user_numeric_features'],
    categorical_features = feature_catalog['user_cat_features'],
    categorical_data = unique_values_df,
    embedding_dim=16,
    depth=4,
    heads=8,
    attn_dropout=0.2,
    ff_dropout=0.2,
    explainable=False,
)

#user encoder
utm_linear_encoder = FTTransformerEncoder(
    numerical_features = feature_catalog['utm_numeric_features'],
    categorical_features = feature_catalog['utm_cat_features'],
    categorical_data = unique_values_df,
    embedding_dim=16,
    depth=4,
    heads=8,
    attn_dropout=0.2,
    ff_dropout=0.2,
    explainable=False,
)



user_model = FTTransformer(
    encoder=user_linear_encoder,
    out_dim=out_dim,
    out_activation=out_activation,
)


utm_model = FTTransformer(
    encoder=utm_linear_encoder,
    out_dim=out_dim,
    out_activation=out_activation,
)


# ####the model

# In[ ]:


metrics = tfrs.metrics.FactorizedTopK(
  candidates = dataset_candidates.batch(100).map(utm_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics,
  num_hard_negatives=10
)   

class NoBaseClassMovielensModel(tf.keras.Model):

  def __init__(self, user_model, utm_model, task):
    super().__init__()
    self.utm_model: tf.keras.Model = utm_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def train_step(self, inputs) -> tf.Tensor:

    # Set up a gradient tape to record gradients.
    with tf.GradientTape() as tape:

      # Loss computation.
      try:
        user_embeddings = self.user_model(inputs)['output']
      except:
          user_embeddings = self.user_model(inputs)

      positive_utm_embeddings = self.utm_model(inputs)
    #   print('user_embeddings: ', user_embeddings)
    #   print('positive_utm_embeddings: ', positive_utm_embeddings)
      loss = self.task(user_embeddings, positive_utm_embeddings,compute_metrics=False)

      # Handle regularization losses as well.
      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, inputs: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    try:
        user_embeddings = self.user_model(inputs)['output']
    except:
        user_embeddings = self.user_model(inputs)
    positive_utm_embeddings = self.utm_model(inputs)
    loss = self.task(user_embeddings, positive_utm_embeddings,compute_metrics=True)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics
# model = NoBaseClassMovielensModel(user_model, utm_model)
# model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


# In[ ]:


model = FTTransformer(
        encoders=encoders,
        utms=utms_dataset_pd,
        out_dim=1,
        out_activation='sigmoid',
    )

optimizer = tf.keras.optimizers.experimental.AdamW(
learning_rate=0.1,
weight_decay=0.000001,
)


model.compile(
    optimizer=optimizer,
    loss={
        "output": tf.keras.losses.BinaryCrossentropy(from_logits=False),
        "importances": None,
    },
    metrics={
        "output": [ 
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Accuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
                    ],
        "importances": None,
    },
)


# ####spark model

# In[ ]:


def run_experiment(params,
    train_set,
    test_set,
    cat_features,
    num_features,
    other_columns,
    target_label,
    encoders,
    train_set_size,
    test_set_size,
    dataset_candidates
    ):

    num_epochs = 2
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']
    train_size = train_set_size
    test_size = test_set_size
    print('hash_key' in train_set.columns)
    print('hash_key' in cat_features)

    converter_train = make_spark_converter(train_set.drop('hash_key'))
    converter_test = make_spark_converter(test_set.drop('hash_key'))

    with converter_train.make_tf_dataset(num_epochs=1, batch_size=batch_size) as dataset_train,\
        converter_test.make_tf_dataset(num_epochs=1, batch_size=batch_size) as dataset_test:

        mapping = create_custom_map_function_petastorm(num_features=num_features, 
                                                        cat_features=cat_features,  
                                                        other_columns = [],                            
                                                        target_label=target_label)

        dataset_train = dataset_train.map(mapping['train'])
        # check.append(dataset_train)
        dataset_test = dataset_test.map(mapping['test'])
        print('finished creating sets')

        metrics = tfrs.metrics.FactorizedTopK(
        candidates = dataset_candidates.batch(100).map(utm_model)
        )

        task = tfrs.tasks.Retrieval(
        metrics=metrics,
        num_hard_negatives = 10
        )   

        model = NoBaseClassMovielensModel(encoders['user_model'], encoders['utm_model'], task)

        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),run_eagerly=True)


        early = EarlyStopping(monitor="val_loss", mode="min", patience=20, restore_best_weights=True)

        run_log_dir = experiment_log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=0,embeddings_freq = 4,write_graph=True,)            

        callback_list = [early, tensorboard_callback]


        print("Start training the model...")        
        print(f'num_epochs {num_epochs}')
        print(f'learning_rate {learning_rate}')
        print(f'weight_decay {weight_decay}')
        print(f'batch_size {batch_size}')
        print(f'train_size {train_size}')
        print(f'train_size {test_size}')

        hist = model.fit(
            dataset_train,
            epochs=params["num_epochs"],
            validation_data=dataset_test,
            callbacks=callback_list,
            steps_per_epoch=train_size//batch_size,
            validation_steps=(test_size // batch_size)
        )
      
        print("Model training finished")
        print(hist.history)
        loss = hist.history['loss'][-1]
        print(f"Validation loss: {round(loss * 100, 2)}%")

    converter_train.delete()
    converter_test.delete()

    return loss, model


# In[ ]:


from hyperopt import hp
from hyperopt import fmin, tpe, Trials
from tensorflow.keras.callbacks import EarlyStopping
global check
check = []
encoders={}
encoders['user_model'] = user_model
encoders['utm_model'] = utm_model

other_columns = ['hash_key'] #['looker_visitor_id', 'utm_id', 'cutoff_date']
# train_data_sample = dtf.toPandas() #needed for the model to determine the feature embeddings values and buckets
train_set, test_set, val_set = df_train.randomSplit([0.8, 0.1, 0.1], seed=42)
# utms_dataset_pd  = df_train.select(utm_cat_cols+utm_numeric_cols).distinct().toPandas()


# ####TRIAN
# 

# In[ ]:


space = {
    'num_epochs': hp.choice('num_epochs', [10, 20]),
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
    'weight_decay': hp.uniform('weight_decay', 0, 0.03),
    'batch_size': hp.choice('batch_size', [128, 512, 1024]),
}
from hyperopt import SparkTrials, STATUS_OK, STATUS_FAIL


from datetime import datetime
train_set=train_set.sample(0.3).coalesce(3)
test_set=test_set.sample(0.3).coalesce(3)
def objective(params):
    accuracy, trained_model = run_experiment(params, 
                                train_set, 
                                test_set, 
                                feature_catalog['user_cat_features']+feature_catalog['utm_cat_features'], 
                                feature_catalog['user_numeric_features']  + feature_catalog['utm_numeric_features'], 
                                feature_catalog['other_columns'], 
                                feature_catalog['label'],
                                encoders,
                                train_set_size,
                                test_set_size,
                                dataset_candidates=dataset_candidates)
    return {'loss': accuracy, 'status': STATUS_OK, 'model': trained_model}

train_set_size=train_set.count()
test_set_size=test_set.count()
print('train_set_size: ', train_set_size)
# Number of evaluations
num_evals = 10

# Trials object to track progress
trials = Trials()

# Run the optimization
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=num_evals,
    trials=trials,
    verbose=2
)
# Find the index of the best trial
best_trial_index = np.argmin(trials.losses())

# Get the best model
best_model = trials.trials[best_trial_index]['result']['model']

print("Best model found:")
print(best_model)

print("Best parameters found:")
print(best_params)


# In[ ]:


train_set.groupby(feature_catalog['label']).count().display()


# In[ ]:


trials.trials[best_trial_index]


# In[ ]:




