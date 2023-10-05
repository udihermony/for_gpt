
class TopK(tf.keras.Model, abc.ABC):
  """Interface for top K layers.

  Implementers must provide the following two methods:

  1. `index`: takes a tensor of candidate embeddings and creates the retrieval
    index.
  2. `call`: takes a tensor of queries and returns top K candidates for those
    queries.
  """

  def __init__(self, k: int, **kwargs) -> None:
    """Initializes the base class."""

    super().__init__(**kwargs)
    self._k = k

  @abc.abstractmethod
  def index(
      self,
      candidates: tf.Tensor,
      identifiers: Optional[tf.Tensor] = None) -> "TopK":
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Matrix of candidate embeddings.
      identifiers: Optional tensor of candidate identifiers. If
        given, these will be used as identifiers of top candidates returned
        when performing searches. If not given, indices into the candidates
        tensor will be returned instead.

    Returns:
      Self.
    """

    raise NotImplementedError()

  def index_from_dataset(
      self,
      candidates: tf.data.Dataset
  ) -> "TopK":
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Dataset of candidate embeddings or (candidate identifier,
        candidate embedding) pairs. If the dataset returns tuples,
        the identifiers will be used as identifiers of top candidates
        returned when performing searches. If not given, indices into the
        candidates dataset will be given instead.

    Returns:
      Self.

    Raises:
      ValueError if the dataset does not have the correct structure.
    """

    _check_candidates_with_identifiers(candidates)

    spec = candidates.element_spec

    if isinstance(spec, tuple):
      identifiers_and_candidates = list(candidates)
      candidates = tf.concat(
          [embeddings for _, embeddings in identifiers_and_candidates],
          axis=0
      )
      identifiers = tf.concat(
          [identifiers for identifiers, _ in identifiers_and_candidates],
          axis=0
      )
    else:
      candidates = tf.concat(list(candidates), axis=0)
      identifiers = None

    return self.index(candidates, identifiers)

  @abc.abstractmethod
  def call(
      self,
      queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      k: Optional[int] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Query the index.

    Args:
      queries: Query features. If `query_model` was provided in the constructor,
        these can be raw query features that will be processed by the query
        model before performing retrieval. If `query_model` was not provided,
        these should be pre-computed query embeddings.
      k: The number of candidates to retrieve. If not supplied, defaults to the
        `k` value supplied in the constructor.

    Returns:
      Tuple of (top candidate scores, top candidate identifiers).

    Raises:
      ValueError if `index` has not been called.
    """

    raise NotImplementedError()

  @tf.function
  def query_with_exclusions(
      self,
      queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      exclusions: tf.Tensor,
      k: Optional[int] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Query the index.

    Args:
      queries: Query features. If `query_model` was provided in the constructor,
        these can be raw query features that will be processed by the query
        model before performing retrieval. If `query_model` was not provided,
        these should be pre-computed query embeddings.
      exclusions: `[query_batch_size, num_to_exclude]` tensor of identifiers to
        be excluded from the top-k calculation. This is most commonly used to
        exclude previously seen candidates from retrieval. For example, if a
        user has already seen items with ids "42" and "43", you could set
        exclude to `[["42", "43"]]`.
      k: The number of candidates to retrieve. Defaults to constructor `k`
        parameter if not supplied.

    Returns:
      Tuple of (top candidate scores, top candidate identifiers).

    Raises:
      ValueError if `index` has not been called.
      ValueError if `queries` is not a tensor (after being passed through
        the query model).
    """

    # Ideally, `exclusions` would simply be an optional parameter to
    # `call`. However, Keras is unable to handle `call` signatures
    # that have more than one Tensor input parameter. The alternative
    # is to either pack all inputs into the first positional argument
    # (via tuples or dicts), or else have a separate method. We opt
    # for the second solution here. The ergonomics in either case aren't
    # great, but having two methods is simpler to explain.
    # See https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/
    # python/keras/engine/base_layer.py#L942 for details of why Keras
    # puts us in this predicament.

    k = k if k is not None else self._k

    adjusted_k = k + exclusions.shape[1]
    x, y = self(queries=queries, k=adjusted_k)
    return _exclude(x, y, exclude=exclusions, k=k)

  @abc.abstractmethod
  def is_exact(self) -> bool:
    """Indicates whether the results returned by the layer are exact.

    Some layers may return approximate scores: for example, the ScaNN layer
    may return approximate results.

    Returns:
      True if the layer returns exact results, and False otherwise.
    """

    raise NotImplementedError()

  def _reset_tf_function_cache(self):
    """Resets the tf.function cache.

    We need to invalidate the compiled tf.function cache here. We just
    dropped some variables and created new ones. The concrete function is
    still referring to the old ones - and because it only holds weak
    references, this does not prevent the old variables being garbage
    collected. The end result is that it references dead objects.
    To resolve this, we throw away the existing tf.function object and
    create a new one.
    """

    if hasattr(self.query_with_exclusions, "python_function"):
      self.query_with_exclusions = tf.function(
          self.query_with_exclusions.python_function)

  def _compute_score(self, queries: tf.Tensor,
                     candidates: tf.Tensor) -> tf.Tensor:
    """Computes the standard dot product score from queries and candidates.

    Args:
      queries: Tensor of queries for which the candidates are to be retrieved.
      candidates: Tensor of candidate embeddings.

    Returns:
      The dot product of queries and candidates.
    """

    return tf.matmul(queries, candidates, transpose_b=True)




"""Layers related to loss computation."""
from typing import Tuple

import numpy as np
import tensorflow as tf

MAX_FLOAT = np.finfo(np.float32).max / 100.0
MIN_FLOAT = np.finfo(np.float32).min / 100.0


def _gather_elements_along_row(data: tf.Tensor,
                               column_indices: tf.Tensor) -> tf.Tensor:
  """Gathers elements from a 2D tensor given the column indices of each row.

  A more efficient way of gathering elements from 2D tensor than tf.gather_nd().
  First, gets the flat 1D indices to gather from. Then flattens the data to 1D
  and uses tf.gather() to generate 1D output and finnally reshapes the
  output back to 2D.

  Args:
    data: A [N, M] 2D `Tensor`.
    column_indices: A [N, K] 2D `Tensor` denoting for each row, the K column
      indices to gather elements from the data `Tensor`.

  Returns:
    A [N, K] `Tensor` including output elements gathered from data `Tensor`.

  Raises:
    ValueError: if the first dimensions of data and column_indices don't match.
  """
  with tf.control_dependencies(
      [tf.assert_equal(tf.shape(data)[0], tf.shape(column_indices)[0])]):
    num_row = tf.shape(data)[0]
    num_column = tf.shape(data)[1]
    num_gathered = tf.shape(column_indices)[1]
    row_indices = tf.tile(
        tf.expand_dims(tf.range(num_row), -1),
        [1, num_gathered])
    flat_data = tf.reshape(data, [-1])
    flat_indices = tf.reshape(
        row_indices * num_column + column_indices, [-1])
    return tf.reshape(
        tf.gather(flat_data, flat_indices), [num_row, num_gathered])


class HardNegativeMining(tf.keras.layers.Layer):
  """Transforms logits and labels to return hard negatives."""

  def __init__(self, num_hard_negatives: int) -> None:
    """Initializes the layer.

    Args:
      num_hard_negatives: How many hard negatives to return.
    """

    super(HardNegativeMining, self).__init__()
    self._num_hard_negatives = num_hard_negatives

  def call(self, logits: tf.Tensor,
           labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Filters logits and labels with per-query hard negative mining.

    The result will include logits and labels for num_hard_negatives
    negatives as well as the positive candidate.

    Args:
      logits: [batch_size, number_of_candidates] tensor of logits.
      labels: [batch_size, number_of_candidates] one-hot tensor of labels.

    Returns:
      logits: [batch_size, num_hard_negatives + 1] tensor of logits.
      labels: [batch_size, num_hard_negatives + 1] one-hot tensor of labels.
    """

    # Number of sampled logits, i.e, the number of hard negatives to be
    # sampled (k) + number of true logit (1) per query, capped by batch size.
    num_sampled = tf.minimum(self._num_hard_negatives + 1, tf.shape(logits)[1])
    # To gather indices of top k negative logits per row (query) in
    # logits, true logits need to be excluded. First replace the true
    # logits (corresponding to positive labels) with a large score value
    # and then select the top k + 1 logits from each
    # row so that selected indices include the indices of true logit + top k
    # negative logits. This approach is to avoid using inefficient
    # tf.boolean_mask() when excluding true logits.

    # For each query, get the indices of the logits which have the highest
    # k + 1 logit values, including the highest k negative logits and one true
    # logit.
    _, col_indices = tf.nn.top_k(
        logits + labels * MAX_FLOAT, k=num_sampled, sorted=False)

    # Gather sampled logits and corresponding labels.
    logits = _gather_elements_along_row(logits, col_indices)
    labels = _gather_elements_along_row(labels, col_indices)

    return logits, labels


class RemoveAccidentalHits(tf.keras.layers.Layer):
  """Zeroes the logits of accidental negatives."""

  def call(self, labels: tf.Tensor, logits: tf.Tensor,
           candidate_ids: tf.Tensor) -> tf.Tensor:
    """Zeros selected logits.

    For each row in the batch, zeros the logits of negative candidates that have
    the same id as the positive candidate in that row.

    Args:
      labels: [batch_size, num_candidates] one-hot labels tensor.
      logits: [batch_size, num_candidates] logits tensor.
      candidate_ids: [num_candidates] candidate identifiers tensor

    Returns:
      logits: Modified logits.
    """
    # A more principled way is to implement softmax_cross_entropy_with_logits
    # with a input mask. Here we approximate so by letting accidental hits
    # have extremely small logits (MIN_FLOAT) for ease-of-implementation.

    candidate_ids = tf.expand_dims(candidate_ids, 1)

    positive_indices = tf.math.argmax(labels, axis=1)
    positive_candidate_ids = tf.gather(candidate_ids, positive_indices)

    duplicate = tf.cast(
        tf.equal(positive_candidate_ids, tf.transpose(candidate_ids)),
        labels.dtype
    )
    duplicate = duplicate - labels

    return logits + duplicate * MIN_FLOAT


class SamplingProbablityCorrection(tf.keras.layers.Layer):
  """Sampling probability correction."""

  def __call__(self, logits: tf.Tensor,
               candidate_sampling_probability: tf.Tensor) -> tf.Tensor:
    """Corrects the input logits to account for candidate sampling probability."""

    return logits - tf.math.log(
        tf.clip_by_value(candidate_sampling_probability, 1e-6, 1.))



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
