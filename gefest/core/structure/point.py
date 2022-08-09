from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    """
    Main class for FEDOT API.
    
    Facade for ApiDataProcessor, ApiComposer, ApiMetrics, ApiInitialAssumptions.

    Args:
        problem: the name of modelling problem to solve

            .. details:: possible ``problem`` options:

                - ``classification`` -> for classification task
                - ``regression`` -> for regression task
                - ``ts_forecasting`` -> for time serires forecasting task

        timeout: time for model design (in minutes): ``None`` or ``-1`` means infinite time
        task_params: additional parameters of the task
        seed: value for fixed random seed
        logging_level: logging levels are the same as in 'logging'

            .. details:: possible ``logging_level`` options:

                    - ``50`` -> critical
                    - ``40`` -> error
                    - ``30`` -> warning
                    - ``20`` -> info
                    - ``10`` -> debug
                    - ``0`` -> nonset
                *Logs with a level HIGHER than set will be displayed*

        safe_mode: if set ``True`` it will cut large datasets to prevent memory overflow and use label encoder
            instead of oneHot encoder if summary cardinality of categorical features is high.
        n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's)
        max_depth: max depth of the pipeline
        max_arity: max arity of the pipeline nodes
        pop_size: population size for composer
        num_of_generations: number of generations for composer
        keep_n_best: Number of the best individuals of previous generation to keep in next generation.
        available_operations: list of model names to use
        stopping_after_n_generation': composer will stop after ``n`` generation without improving
        with_tuning: allow hyperparameters tuning for the model
        cv_folds: number of folds for cross-validation
        validation_blocks: number of validation blocks for time series forecasting
        max_pipeline_fit_time: time constraint for operation fitting (minutes)
        initial_assumption: initial assumption for composer
        genetic_scheme: name of the genetic scheme
        history_folder: name of the folder for composing history
        composer_metric:  metric for quality calculation during composing
        collect_intermediate_metric: save metrics for intermediate (non-root) nodes in pipeline
        preset: name of preset for model building (e.g. 'best_quality', 'fast_train', 'gpu'):

            .. details:: possible ``preset`` options:

                - ``best_quality`` -> All models that are available for this data type and task are used
                - ``fast_train`` -> Models that learn quickly. This includes preprocessing operations
                  (data operations) that only reduce the dimensionality of the data, but cannot increase it.
                  For example, there are no polynomial features and one-hot encoding operations
                - ``stable`` -> The most reliable preset in which the most stable operations are included.
                - ``auto`` -> Automatically determine which preset should be used.
                - ``gpu`` -> Models that use GPU resources for computation.
                - ``ts`` -> A special preset with models for time series forecasting task.
                - ``automl`` -> A special preset with only AutoML libraries such as TPOT and H2O as operations.
                - ``*tree`` -> A special preset that allows only tree-based algorithms

        tuner_metric:  metric for quality calculation during tuning
        use_pipelines_cache: bool indicating whether to use pipeline structures caching, enabled by default.
        use_preprocessing_cache: bool indicating whether to use optional preprocessors caching, enabled by default.
  
    """

    _x: float
    _y: float
    _z: float = 0.0

    @property
    def x(self) -> int:
        return round(self._x)

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self) -> int:
        return round(self._y)

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def z(self) -> int:
        return round(self._z)

    @z.setter
    def z(self, value):
        self._z = value

    def coords(self):
        '''Returns the :obj:`list` included spatial coordinates of the :obj:`Point`

        Returns:
          :obj:`List`: ``[x,y,z]``

        '''
        return [self.x, self.y, self.z]


@dataclass
class Point2D(Point):
    @property
    def z(self):
        return 0

    @z.setter
    def z(self, value):
        self._z = 0

    def coords(self):
        return [self.x, self.y]
