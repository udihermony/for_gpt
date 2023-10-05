

class Streaming(TopK):
  """Retrieves K highest scoring items and their ids from a large dataset.

  Used to efficiently retrieve top K query-candidate scores from a dataset,
  along with the top scoring candidates' identifiers.
  """

  def __init__(self,
               query_model: Optional[tf.keras.Model] = None,
               k: int = 10,
               handle_incomplete_batches: bool = True,
               num_parallel_calls: int = tf.data.AUTOTUNE,
               sorted_order: bool = True) -> None:
    """Initializes the layer.

    Args:
      query_model: Optional Keras model for representing queries. If provided,
        will be used to transform raw features into query embeddings when
        querying the layer. If not provided, the layer will expect to be given
        query embeddings as inputs.
      k: Number of top scores to retrieve.
      handle_incomplete_batches: When True, candidate batches smaller than k
        will be correctly handled at the price of some performance. As an
        alternative, consider using the drop_remainer option when batching the
        candidate dataset.
      num_parallel_calls: Degree of parallelism when computing scores. Defaults
        to autotuning.
      sorted_order: If the resulting scores should be returned in sorted order.
        setting this to False may result in a small increase in performance.

    Raises:
      ValueError if candidate elements are not tuples.
    """

    super().__init__(k=k)

    self.query_model = query_model
    self._candidates = None
    self._handle_incomplete_batches = handle_incomplete_batches
    self._num_parallel_calls = num_parallel_calls
    self._sorted = sorted_order

    self._counter = self.add_weight("counter", dtype=tf.int32, trainable=False)

  def index_from_dataset(
      self,
      candidates: tf.data.Dataset
  ) -> "TopK":

    _check_candidates_with_identifiers(candidates)

    self._candidates = candidates

    return self

  def index(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      candidates: tf.data.Dataset,
      identifiers: Optional[tf.data.Dataset] = None) -> "Streaming":
    """Not implemented. Please call `index_from_dataset` instead."""

    raise NotImplementedError(
        "The streaming top k class only accepts datasets. "
        "Please call `index_from_dataset` instead."
    )

  def call(
      self,
      queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      k: Optional[int] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:

    k = k if k is not None else self._k

    if self._candidates is None:
      raise ValueError("The `index` method must be called first to "
                       "create the retrieval index.")

    if self.query_model is not None:
      queries = self.query_model(queries)

    # Reset the element counter.
    self._counter.assign(0)

    def top_scores(candidate_index: tf.Tensor,
                   candidate_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Computes top scores and indices for a batch of candidates."""

      scores = self._compute_score(queries, candidate_batch)

      if self._handle_incomplete_batches:
        k_ = tf.math.minimum(k, tf.shape(scores)[1])
      else:
        k_ = k

      scores, indices = tf.math.top_k(scores, k=k_, sorted=self._sorted)

      return scores, tf.gather(candidate_index, indices)

    def top_k(state: Tuple[tf.Tensor, tf.Tensor],
              x: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
      """Reduction function.

      Returns top K scores from a combination of existing top K scores and new
      candidate scores, as well as their corresponding indices.

      Args:
        state: tuple of [query_batch_size, k] tensor of highest scores so far
          and [query_batch_size, k] tensor of indices of highest scoring
          elements.
        x: tuple of [query_batch_size, k] tensor of new scores and
          [query_batch_size, k] tensor of new indices.

      Returns:
        Tuple of [query_batch_size, k] tensors of highest scores and indices
          from state and x.
      """
      state_scores, state_indices = state
      x_scores, x_indices = x

      joined_scores = tf.concat([state_scores, x_scores], axis=1)
      joined_indices = tf.concat([state_indices, x_indices], axis=1)

      if self._handle_incomplete_batches:
        k_ = tf.math.minimum(k, tf.shape(joined_scores)[1])
      else:
        k_ = k

      scores, indices = tf.math.top_k(joined_scores, k=k_, sorted=self._sorted)

      return scores, tf.gather(joined_indices, indices, batch_dims=1)

    def enumerate_rows(batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Enumerates rows in each batch using a total element counter."""

      starting_counter = self._counter.read_value()
      end_counter = self._counter.assign_add(tf.shape(batch)[0])

      return tf.range(starting_counter, end_counter), batch

    if not isinstance(self._candidates.element_spec, tuple):
      # We don't have identifiers.
      candidates = self._candidates.map(enumerate_rows)
      index_dtype = tf.int32
    else:
      candidates = self._candidates
      index_dtype = self._candidates.element_spec[0].dtype

    # Initialize the state with dummy scores and candidate indices.
    initial_state = (tf.zeros((tf.shape(queries)[0], 0), dtype=tf.float32),
                     tf.zeros((tf.shape(queries)[0], 0), dtype=index_dtype))

    with _wrap_batch_too_small_error(k):
      results = (
          candidates
          # Compute scores over all candidates, and select top k in each batch.
          # Each element is a ([query_batch_size, k] tensor,
          # [query_batch_size, k] tensor) of scores and indices (where query_
          # batch_size is the leading dimension of the input query embeddings).
          .map(top_scores, num_parallel_calls=self._num_parallel_calls)
          # Reduce into a single tuple of output tensors by keeping a running
          # tally of top k scores and indices.
          .reduce(initial_state, top_k))

    return results

  def is_exact(self) -> bool:
    return True



class FactorizedTopK(Factorized):
  """Computes metrics for across top K candidates surfaced by a retrieval model.

  The default metric is top K categorical accuracy: how often the true candidate
   is in the top K candidates for a given query.
  """

  def __init__(
      self,
      candidates: Union[layers.factorized_top_k.TopK, tf.data.Dataset],
      ks: Sequence[int] = (1, 5, 10, 50, 100),
      name: str = "factorized_top_k",
  ) -> None:
    """Initializes the metric.

    Args:
      candidates: A layer for retrieving top candidates in response
        to a query, or a dataset of candidate embeddings from which
        candidates should be retrieved.
      ks: A sequence of values of `k` at which to perform retrieval evaluation.
      name: Optional name.
    """

    super().__init__(name=name)

    if isinstance(candidates, tf.data.Dataset):
      candidates = (
          layers.factorized_top_k.Streaming(k=max(ks))
          .index_from_dataset(candidates)
      )

    self._ks = ks
    self._candidates = candidates
    self._top_k_metrics = [
        tf.keras.metrics.Mean(
            name=f"{self.name}/top_{x}_categorical_accuracy"
        ) for x in ks
    ]

  def update_state(
      self,
      query_embeddings: tf.Tensor,
      true_candidate_embeddings: tf.Tensor,
      true_candidate_ids: Optional[tf.Tensor] = None,
      sample_weight: Optional[tf.Tensor] = None,
  ) -> tf.Operation:
    """Updates the metrics.

    Args:
      query_embeddings: [num_queries, embedding_dim] tensor of query embeddings.
      true_candidate_embeddings: [num_queries, embedding_dim] tensor of
        embeddings for candidates that were selected for the query.
      true_candidate_ids: Ids of the true candidates. If supplied, evaluation
        will be id-based: the supplied ids will be matched against the ids of
        the top candidates returned from the retrieval index, which should have
        been constructed with the appropriate identifiers.

        If not supplied, evaluation will be score-based: the score of the true
        candidate will be computed and compared with the scores returned from
        the index for the top candidates.

        Score-based evaluation is useful for when the true candidate is not
        in the retrieval index. Id-based evaluation is useful for when scores
        returned from the index are not directly comparable to scores computed
        by multiplying the candidate and embedding vector. For example, scores
        returned by ScaNN are quantized, and cannot be compared to
        full-precision scores.
      sample_weight: Optional weighting of each example. Defaults to 1.

    Returns:
      Update op. Only used in graph mode.
    """

    if true_candidate_ids is None and not self._candidates.is_exact():
      raise ValueError(
          f"The candidate generation layer ({self._candidates}) does not return "
          "exact results. To perform evaluation using that layer, you must "
          "supply `true_candidate_ids`, which will be checked against "
          "the candidate ids returned from the candidate generation layer."
      )

    positive_scores = tf.reduce_sum(
        query_embeddings * true_candidate_embeddings, axis=1, keepdims=True)

    top_k_predictions, retrieved_ids = self._candidates(
        query_embeddings, k=max(self._ks))

    update_ops = []

    if true_candidate_ids is not None:
      # We're using ID-based evaluation.
      if len(true_candidate_ids.shape) == 1:
        true_candidate_ids = tf.expand_dims(true_candidate_ids, 1)

      # Deal with ScaNN using `NaN`-padding by converting its
      # `NaN` scores into minimum scores.
      nan_padding = tf.math.is_nan(top_k_predictions)
      top_k_predictions = tf.where(
          nan_padding,
          tf.ones_like(top_k_predictions) * tf.float32.min,
          top_k_predictions
      )

      # Check sortedness.
      is_sorted = (
          top_k_predictions[:, :-1] - top_k_predictions[:, 1:]
      )
      tf.debugging.assert_non_negative(
          is_sorted, message="Top-K predictions must be sorted."
      )

      # Check whether the true candidates were retrieved, accounting
      # for padding.
      ids_match = tf.cast(
          tf.math.logical_and(
              tf.math.equal(true_candidate_ids, retrieved_ids),
              tf.math.logical_not(nan_padding)
          ),
          tf.float32
      )

      for k, metric in zip(self._ks, self._top_k_metrics):
        # By slicing until :k we assume scores are sorted.
        # Clip to only count multiple matches once.
        match_found = tf.clip_by_value(
            tf.reduce_sum(ids_match[:, :k], axis=1, keepdims=True),
            0.0, 1.0
        )
        update_ops.append(metric.update_state(match_found, sample_weight))
    else:
      # Score-based evaluation.
      y_pred = tf.concat([positive_scores, top_k_predictions], axis=1)

      for k, metric in zip(self._ks, self._top_k_metrics):
        targets = tf.zeros(tf.shape(positive_scores)[0], dtype=tf.int32)
        top_k_accuracy = tf.math.in_top_k(
            targets=targets,
            predictions=y_pred,
            k=k
        )
        update_ops.append(metric.update_state(top_k_accuracy, sample_weight))

    return tf.group(update_ops)





class Retrieval(tf.keras.layers.Layer, base.Task):
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
    """Initializes the task.

    Args:
      loss: Loss function. Defaults to
        `tf.keras.losses.CategoricalCrossentropy`.
      metrics: Object for evaluating top-K metrics over a
       corpus of candidates. These metrics measure how good the model is at
       picking the true candidate out of all possible candidates in the system.
       Note, because the metrics range over the entire candidate set, they are
       usually much slower to compute. Consider setting `compute_metrics=False`
       during training to save the time in computing the metrics.
      batch_metrics: Metrics measuring how good the model is at picking out the
       true candidate for a query from other candidates in the batch. For
       example, a batch AUC metric would measure the probability that the true
       candidate is scored higher than the other candidates in the batch.
      loss_metrics: List of Keras metrics used to summarize the loss.
      temperature: Temperature of the softmax.
      num_hard_negatives: If positive, the `num_hard_negatives` negative
        examples with largest logits are kept when computing cross-entropy loss.
        If larger than batch size or non-positive, all the negative examples are
        kept.
      remove_accidental_hits: When given
        enables removing accidental hits of examples used as negatives. An
        accidental hit is defined as a candidate that is used as an in-batch
        negative but has the same id with the positive candidate.
      name: Optional task name.
    """

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

  def call(self,
           query_embeddings: tf.Tensor,
           candidate_embeddings: tf.Tensor,
           sample_weight: Optional[tf.Tensor] = None,
           candidate_sampling_probability: Optional[tf.Tensor] = None,
           candidate_ids: Optional[tf.Tensor] = None,
           compute_metrics: bool = True,
           compute_batch_metrics: bool = True) -> tf.Tensor:
    """Computes the task loss and metrics.

    The main argument are pairs of query and candidate embeddings: the first row
    of query_embeddings denotes a query for which the candidate from the first
    row of candidate embeddings was selected by the user.

    The task will try to maximize the affinity of these query, candidate pairs
    while minimizing the affinity between the query and candidates belonging
    to other queries in the batch.

    Args:
      query_embeddings: [num_queries, embedding_dim] tensor of query
        representations.
      candidate_embeddings: [num_candidates, embedding_dim] tensor of candidate
        representations. Normally, `num_candidates` is the same as
        `num_queries`: there is a positive candidate corresponding for every
        query. However, it is also possible for `num_candidates` to be larger
        than `num_queries`. In this case, the extra candidates will be used an
        extra negatives for all queries.
      sample_weight: [num_queries] tensor of sample weights.
      candidate_sampling_probability: Optional tensor of candidate sampling
        probabilities. When given will be be used to correct the logits to
        reflect the sampling probability of negative candidates.
      candidate_ids: Optional tensor containing candidate ids. When given,
        factorized top-K evaluation will be id-based rather than score-based.
      compute_metrics: Whether to compute metrics. Set this to False
        during training for faster training.
      compute_batch_metrics: Whether to compute batch level metrics.
        In-batch loss_metrics will still be computed.
    Returns:
      loss: Tensor of loss values.
    """

    scores = tf.linalg.matmul(
        query_embeddings, candidate_embeddings, transpose_b=True)

    num_queries = tf.shape(scores)[0]
    num_candidates = tf.shape(scores)[1]

    labels = tf.eye(num_queries, num_candidates)

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

    loss = self._loss(y_true=labels, y_pred=scores, sample_weight=sample_weight)

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
