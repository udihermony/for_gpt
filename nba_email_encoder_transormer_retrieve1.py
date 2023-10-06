

class retrievalModel(tf.keras.Model):

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
      loss = self.task(user_embeddings, positive_utm_embeddings,compute_metrics=True)

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

        dataset_test = dataset_test.map(mapping['test'])
        print('finished creating sets')



        metrics = tfrs.metrics.FactorizedTopK(
        candidates = dataset_candidates.batch(500).map(encoders['utm_model'])
        )
        task = RetrievalCustom(
        metrics=metrics,
        fixed_library_embeddings = dataset_candidates
        )   

        model = retrievalModel(encoders['user_model'], encoders['utm_model'], task)

        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),run_eagerly=True)

        early = EarlyStopping(monitor="val_loss", mode="min", patience=20, restore_best_weights=True)

        run_log_dir = experiment_log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=0,embeddings_freq = 4,write_graph=True,)            

        callback_list = [early, tensorboard_callback]


        hist = model.fit(
            dataset_train,
            epochs=params["num_epochs"],
            validation_data=dataset_test,
            callbacks=callback_list,
            steps_per_epoch=train_size//batch_size,
            validation_steps=(test_size // batch_size)
        )
     
        loss = hist.history['loss'][-1]
        print(f"Validation loss: {round(loss * 100, 2)}%")

    converter_train.delete()
    converter_test.delete()

    return loss, model


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
    explainable=True,
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


space = {
    'num_epochs': hp.choice('num_epochs', [2, 4]),
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
    'weight_decay': hp.uniform('weight_decay', 0, 0.03),
    'batch_size': hp.choice('batch_size', [10, 50, 1000]),
}
from hyperopt import SparkTrials, STATUS_OK, STATUS_FAIL


from datetime import datetime

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
train_set = train_set.coalesce(2)
train_set_size=train_set.count()
test_set_size=test_set.count()
print('train_set_size: ', train_set_size)
# Number of evaluations
num_evals = 2

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



class RetrievalCustom(tf.keras.layers.Layer, base.Task):
  """A factorized retrieval task.

  Recommender systems are often composed of two components:
  - a retrieval model, retrieving O(thousands) candidates from a corpus of
    O(millions) candidates.
  - a ranker model, scoring the candidates retrieved by the retrieval model to
    return a ranked shortlist of a few dozen candidates.

  This task defines models that facilitate efficient retrieval of candidates
  from large corpora by maintaining a two-tower, factorized structure: separate
  query and candidate representation towers, joined at the top via a lightweight
  scoring function.
  """

  def __init__(self,
               fixed_library_dataset=None,
               loss: Optional[tf.keras.losses.Loss] = None,
               metrics: Optional[Union[
                   Sequence[tfrs_metrics.Factorized],
                   tfrs_metrics.Factorized
               ]] = None,
               batch_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
               loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
               temperature: Optional[float] = None,
               num_hard_negatives: Optional[int] = None,
               remove_accidental_hits: bool = False,
               name: Optional[Text] = None) -> None:
  

    super().__init__(name=name)

    self._loss = loss if loss is not None else tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    if metrics is None:
      metrics = []
    if not isinstance(metrics, Sequence):
      metrics = [metrics]

    self._factorized_metrics = metrics
    self._batch_metrics = batch_metrics or []
    self._loss_metrics = loss_metrics or []
    self._temperature = temperature
    self._num_hard_negatives = num_hard_negatives
    self._remove_accidental_hits = remove_accidental_hits
    self._fixed_library_dataset = fixed_library_dataset

  @property
  def factorized_metrics(self) -> Optional[
      Sequence[tfrs_metrics.Factorized]]:
    """The metrics object used to compute retrieval metrics."""

    return self._factorized_metrics

  @factorized_metrics.setter
  def factorized_metrics(self,
                         value: Optional[Union[
                             Sequence[tfrs_metrics.Factorized],
                             tfrs_metrics.Factorized
                         ]]) -> None:
    """Sets factorized metrics."""

    if not isinstance(value, Sequence):
      value = []

    self._factorized_metrics = value
  def _get_labels_for_fixed_library(self, candidate_embeddings):
    # Compute similarity between batch candidate embeddings and fixed library embeddings
    similarity = tf.linalg.matmul(candidate_embeddings, self.fixed_library_embeddings_materialized, transpose_b=True)
    
    # For each query, find the index of its true candidate in the fixed library
    indices = tf.math.argmax(similarity, axis=1)

    # Construct the labels matrix
    labels = tf.one_hot(indices, depth=tf.shape(self.fixed_library_embeddings_materialized)[0])

    return labels
  def call(self,utm_model_encoder,
           query_embeddings: tf.Tensor,
           candidate_embeddings: tf.Tensor,
           sample_weight: Optional[tf.Tensor] = None,
           candidate_sampling_probability: Optional[tf.Tensor] = None,
           candidate_ids: Optional[tf.Tensor] = None,
           compute_metrics: bool = True,
           compute_batch_metrics: bool = True ) -> tf.Tensor:
  
    
    self.fixed_library_embeddings_materialized = next(iter(self._fixed_library_dataset.batch(500).map(utm_model)))

    # Compute similarity with the fixed library
    similarity_with_library = tf.linalg.matmul(query_embeddings, self.fixed_library_embeddings_materialized, transpose_b=True)
    # print('similarity_with_library ', similarity_with_library.shape())

    # Get the labels for the fixed library based on the true candidates from the batch
    labels_for_fixed_library = self._get_labels_for_fixed_library(candidate_embeddings)

    # Compute the loss
    loss = self._loss(y_true=labels_for_fixed_library, y_pred=similarity_with_library)

    num_queries = tf.shape(labels_for_fixed_library)[0]
    num_candidates = tf.shape(labels_for_fixed_library)[1]

   
    if self._temperature is not None:
      scores = scores / self._temperature

    if candidate_sampling_probability is not None:
      scores = layers.loss.SamplingProbablityCorrection()(
          scores, candidate_sampling_probability)

    if self._remove_accidental_hits:
      if candidate_ids is None:
        raise ValueError(
            "When accidental hit removal is enabled, candidate ids "
            "must be supplied."
        )
      scores = layers.loss.RemoveAccidentalHits()(labels, scores, candidate_ids)

    if self._num_hard_negatives is not None:
      scores, labels = layers.loss.HardNegativeMining(self._num_hard_negatives)(
          scores,
          labels)

   update_ops = []
    for metric in self._loss_metrics:
      update_ops.append(
          metric.update_state(loss, sample_weight=sample_weight))

    if compute_metrics:
      for metric in self._factorized_metrics:
        update_ops.append(
            metric.update_state(
                query_embeddings,
                # Slice to the size of query embeddings
                # if `candidate_embeddings` contains extra negatives.
                candidate_embeddings[:tf.shape(query_embeddings)[0]],
                true_candidate_ids=candidate_ids)
        )

    if compute_batch_metrics:
      for metric in self._batch_metrics:
        update_ops.append(metric.update_state(labels, scores))

    with tf.control_dependencies(update_ops):
      return tf.identity(loss)

    

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
