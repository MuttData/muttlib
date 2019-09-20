"""Abstract machine learning experiments and/or program runs."""
from pprint import pformat
from textwrap import dedent
import logging
from pathlib import PosixPath
import pandas as pd
import mlflow

from muttlib.dbconn import session_scope
from muttlib.utils import hash_str
from muttlib.gcd import AttributeHelperMixin

logger = logging.getLogger(f'runs.{__name__}')


class BaseRun(AttributeHelperMixin):
    """Implement abstract runs abstraction.

    Subclasses should implement/define schema and output model classes (required, for
    instance, to get a proper run_id).
    """

    VALID_ENVS = ['dev', 'stg', 'prd']

    def __init__(
        self,
        time_range_conf,
        cli_args,
        env,
        confs_paths_d,
        name=None,
        do_plots=True,
        dry_run=False,
        use_mlflow=True,
    ):
        """
        Construct BaseRun object.

        Args:
            time_range_conf (gcd.TimeRangeConfiguration): an instance of a
                time-range-conf.
            cli_args (dict): Map of command-line-argument name to value.
            confs_paths_d (dict): Map of modules to their config files.

        Notes:
            Not nice but removes the need of cooperation from subclasses.

        """
        self.time_range_conf = time_range_conf
        self.cli_args = cli_args
        self.do_plots = do_plots
        self.dry_run = dry_run
        self.use_mlflow = use_mlflow
        self.env = env
        if name is None:
            name = f'{env}-{self.__class__.__name__}'
        self.name = name

        self.created_at = pd.datetime.utcnow().replace(microsecond=0)
        confs_paths_d.update(dict(environment=self.env))
        self.confs = confs_paths_d

        # Empty at first attrs
        self.pipeline_steps = []
        self.row_instance = None
        self.run_id = None
        self._validate_attrs_types()
        self._validate_attrs_values()

    def _check_type_attrs_l(self, typ, l):
        """Check if list of object's attrs-names corresponds to passed type."""
        typ_check = [isinstance(getattr(self, attr), typ) for attr in l]
        if not all(typ_check):
            raise TypeError(  # type: ignore
                f"Some attrs expected to be `{typ}`-typed were not found to be so: "
                f" Attrs:{l}, Checks:{typ_check}."
            )

    def _validate_attrs_types(self):
        """Validate construction attrs types."""
        self._check_type_attrs_l(dict, ['cli_args', 'confs'])
        self._check_type_attrs_l(bool, ['do_plots', 'dry_run', 'use_mlflow'])
        self._check_type_attrs_l(str, ['env', 'name'])

    def _validate_attrs_values(self):
        """Validate construction attrs values."""
        if self.env not in self.VALID_ENVS:
            raise ValueError(  # type: ignore
                f'Bad `env` passed: {self.env}. Valid values: {self.VALID_ENVS}'
            )

        if not all(
            isinstance(self.confs[k], str) or isinstance(self.confs[k], PosixPath)
            for k in self.confs
        ):
            raise ValueError(  # type: ignore
                f"Bad `conf` dict passed: {self.confs}."
                f"Dict can hold Path or string values only."
            )

    @property
    def schema_cls(self):  # noqa: D102
        raise NotImplementedError("Subclass should implement this.")

    @property
    def out_schema_cls(self):  # noqa: D102
        raise NotImplementedError("Subclass should implement this.")

    def start_run(self):
        """Do setup of everything needed to start the run."""
        logger.info(f"Starting {self.__class__.__name__} run...")
        logger.info(f"Run params:\n {self.get_summary()}")

        if self.use_mlflow:
            self._init_mlflow()

    def _init_mlflow(self):
        mlflow.start_run(run_name=self.name)
        log_items = [self.time_range_conf] + self.pipeline_steps
        for s in log_items:
            for k, v in s.get_params().items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        mlflow.log_param(f"{s.__class__.__name__}.{k}.{k2}", v2)
                else:
                    mlflow.log_param(f"{s.__class__.__name__}.{k}", v)

    def _collect_step_data_to_mlflow(self):
        for s in self.pipeline_steps:
            for p in s.get_artifacts():
                mlflow.log_artifact(p)
            metrics = {
                f"{s.__class__.__name__}.{k}": v for k, v in s.get_metrics().items()
            }
            mlflow.log_metrics(metrics)

    def _build_repr_section(self, title, lines):
        msg_parts = [f' {title} '.center(40, '#')] + lines
        msg = '\n'.join(msg_parts) + '\n'
        return msg

    def get_summary(self):
        """Print pretty representations of params objects as strings."""
        # noqa: D202
        def _std_section_formatter(title, items):
            lines = list(map(pformat, items))
            msg = self._build_repr_section(title, lines)
            return msg

        # Run confs
        run_conf_msg = _std_section_formatter('RUN CONFS', [self, self.confs])

        # Time range confs
        time_range_msg = _std_section_formatter('TIME RANGE', [self.time_range_conf])

        # Parameter confs
        step_params_msg = _std_section_formatter('STEP PARAMS', self.pipeline_steps)

        # Build message showing CLI arguments.
        cli_args_lines = [f'{k}={v}' for k, v in self.cli_args.items()]
        cli_args_msg = self._build_repr_section('CLI ARGS', cli_args_lines)

        rv = "\n\n".join([run_conf_msg, time_range_msg, step_params_msg, cli_args_msg])
        rv = dedent("<\n" f"{rv}" "\n>")
        return rv

    def get_hash(self):
        """Get identifier hash."""
        hash_items = [self, self.confs, self.time_range_conf, self.pipeline_steps]
        s = '\n'.join(map(repr, hash_items)) + '\n'
        rv = hash_str(s)
        return rv

    def _prepare_output_data(self):
        """Create run's out dict.

        This data represents a row of the corresponding schema_cls.
        It will be used to instantiate the run's id.
        """
        rv = {
            'name': self.name,
            'start_date': self.time_range_conf.start_date,
            'end_date': self.time_range_conf.end_date,
            'future_date': self.time_range_conf.future_date,
            'params': self.get_summary(),
            'created_datetime': self.created_at,
            'params_hash': self.get_hash(),
        }
        return rv

    def _save_run(self, sess):
        """Save current run."""
        out_d = self._prepare_output_data()
        self.row_instance = self.schema_cls(**out_d)
        sess.add(self.row_instance)
        sess.flush()
        self.run_id = self.row_instance.id

    def _save_results(self, sess, output_df):
        """Save output results."""
        output_df['run_id'] = self.run_id
        rv = sess.bulk_insert_mappings(
            self.out_schema_cls, output_df.to_dict(orient='records')
        )
        return rv

    def save(self, engine, output_df):
        """Save run and insert results."""
        with session_scope(engine) as sess:
            logger.info(f"Saving run...")
            self._save_run(sess)
            logger.info(f"Saving {len(output_df)} results...")
            self._save_results(sess, output_df)
            logger.info(f"Committing run data...")

    def end_run(self, engine, output_df):
        """Perform steps neccesary to end the run."""
        if self.use_mlflow:
            self._collect_step_data_to_mlflow()
            mlflow.end_run()

        if self.dry_run:
            logger.info(f"Skipping save due to dry_run=True ...")
        else:
            self.save(engine, output_df)

        logger.info(f"Finished {self.__class__.__name__} run_id: {self.run_id}!")
        logger.info(f"All done! Have a nice day ^_^")
