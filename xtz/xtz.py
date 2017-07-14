#pylint: disable=expression-not-assigned, too-many-arguments, too-few-public-methods, C0103
"""xtz helps you build linear pipelines

The xtz module provides an environment to easily inject dependencies, log and debug
pipeline steps.

Example:
    import logging

    from typing import List
    from xtz import Inject, LastInput, Pipeline, pipe

    @pipe
    def first_step(required_param: int,
                logger: logging.Logger=None, # child logger will be created
                db_string: str = None, # will be injected
                config_val: str = Inject("dev.config_value1")): # another value injected
        logger.info('executing first_step %s %s %s', required_param, db_string, config_val)

        return ["output from first_step"]

    @pipe
    def second_step(sumthing,
                    last_input: List = LastInput, # Using 'LastInput' will
                                                # take the value from the previous step
                    logger: logging.Logger=None):
        logger.info('executing second_step %s, %s', sumthing, last_input)

        return ['output from second_step']

    @pipe
    def third_step(last_input: List = LastInput,
                logger: logging.Logger=None):
        logger.info('executing third_step %s', last_input)

        return ['output from third_step']

    class StepInClass:
        def __init__(self):
            pass

        def some_method(self, logger):
            logger.info("calling from some method")

        @pipe
        def fourth_step(self,
                        required_param1, # required to be passed
                        required_param2, # required to be passed
                        last_input: List = LastInput,
                        logger: logging.Logger=None,
                        test: str = None):
            self.some_method(logger)
            logger.info('executing fourth_step %s %s %s %s',
                        required_param1,
                        required_param2,
                        last_input,
                        test)

    def main():
        # setup logging
        logging.basicConfig()
        logging.getLogger().setLevel(logging.WARN)
        logger = logging.getLogger().getChild("main")
        logger.setLevel(logging.DEBUG)

        # initialize some dependencies
        step_in_class = StepInClass()
        test_str = 'variables are made available in the REPL'

        # create the pipeline with things to inject
        pipeline = Pipeline(inject={'db_string': 'jdbc://test',
                                    'dev.config_value1':'Hello!'},
                            logger=logger) # loggers will be created as child loggers for each step

        # start recording calls and passed parameters
        pipeline.record()

        first_step(1)
        second_step(2)
        third_step()
        step_in_class.fourth_step(1, # multiline steps are supported in REPL
                                2,
                                test=test_str)

        # execute the pipeline
        pipeline.execute(interactive=False) # set interactive=True to drop into REPL

    if __name__ == '__main__':
        main()

Executing:
    > python test.py
    INFO:main:Executing 4 steps
    INFO:main:      1 'first_step'
    INFO:main:      2 'second_step'
    INFO:main:      3 'third_step'
    INFO:main:      4 'fourth_step'
    INFO:main:>>> Beginning pipeline execution...
    INFO:main.first_step:>> Starting step 1 of 4
    INFO:main.first_step:executing first_step 1 jdbc://test Hello!
    INFO:main.first_step:>> Finished step 1 of 4 in 0:00:00
    INFO:main.second_step:>> Starting step 2 of 4
    INFO:main.second_step:executing second_step 2, ['output from first_step']
    INFO:main.second_step:>> Finished step 2 of 4 in 0:00:00.000500
    INFO:main.third_step:>> Starting step 3 of 4
    INFO:main.third_step:executing third_step ['output from second_step']
    INFO:main.third_step:>> Finished step 3 of 4 in 0:00:00.001000
    INFO:main.fourth_step:>> Starting step 4 of 4
    INFO:main.fourth_step:calling from some method
    INFO:main.fourth_step:executing fourth_step 1 2 ['output from third_step'] None
    INFO:main.fourth_step:>> Finished step 4 of 4 in 0:00:00.000501
    INFO:main.fourth_step:>>> Finished pipeline execution in 0:00:00.002619

Executing (with interactive):
    > python test.py
    Press F3 to quickly execute steps in your pipeline. See 'last_output' to view last output from a step.

    >>> first_step(1)
    ... second_step(2)
    ... third_step()
    ... step_in_class.fourth_step(1, # multiline steps are supported in REPL
    ...                           2,
    ...                           test=test_str)
    INFO:main.first_step:executing first_step 1 jdbc://test Hello!
    INFO:main.second_step:executing second_step 2, ['output from first_step']
    INFO:main.third_step:executing third_step ['output from second_step']
    INFO:main.fourth_step:calling from some method
    INFO:main.fourth_step:executing fourth_step 1 2 ['output from third_step'] None
    Finished first_step in 0:00:00.000500
    Using '['output from first_step']' from 'first_step''
    Finished second_step in 0:00:00.000500
    Using '['output from second_step']' from 'second_step''
    Finished third_step in 0:00:00.000500
    Using '['output from third_step']' from 'third_step''
    Finished fourth_step in 0:00:00.000500
    >>> test_str
    'variables are made available in the REPL'
    >>>

    [F4] Vi (INSERT)   8/8 [F3] History [F6] Paste mode

Todo:
    * Log exceptions better
    * Event hooks
    * Support injecting stuff into class constructors
    * Support async flows for non-linear pipelines
"""
import copy
import datetime
import inspect
import linecache
import logging
import os
import textwrap
import time

import decorator
import ptpython.repl

from typing import Any, List, Mapping, TypeVar


class Inject(str):
    """Use this as a default value to inject values with arbitrary names"""

    def __new__(cls, content):
        o = super(Inject, cls).__new__(cls, content)
        return o


LastInput = TypeVar('LastInput')

class _DeferredPipeCall(object):
    def __init__(self, orig_call_str, func, args, kwargs, log_step=True):
        self.orig_call_str = orig_call_str
        self.func = func
        self.orig_args = args
        self.orig_kwargs = kwargs
        self.rebuilt_args = None
        self.rebuilt_kwargs = None
        self.live_params = None
        self.last_input_param = None
        self.pipeline_param = None
        self.logger_param = None
        self.log_step = log_step

    def inspect(self):
        funcsig = inspect.signature(self.func)
        return funcsig.bind(*self.rebuilt_args, **self.rebuilt_kwargs)

    def bind(self, inject=None, inject_fail_on_none=False):
        if inject is None:
            inject = {}

        # check for unfulfilled arguments
        funcsig = inspect.signature(self.func)
        orig_boundargs = funcsig.bind(*self.orig_args, **self.orig_kwargs)
        orig_boundargs.apply_defaults()

        injecting_kwargs = {}
        rebuilt_args = []
        inject_errors = []

        # bind injectables
        for i, param in enumerate(funcsig.parameters.values()):
            if param.default == inspect.Parameter.empty:  # purely positional argument
                rebuilt_args.append(self.orig_args[i])
            else:
                if isinstance(param.default, Inject):
                    # found a specific string
                    if str(param.default) in inject:
                        injecting_kwargs[param.name] = inject[str(
                            param.default)]
                    elif inject_fail_on_none:
                        inject_errors.append((param.name, str(param.default)))
                elif self.orig_args[i] != None and self.orig_args[i] != LastInput:
                    # it was passed in specfically
                    injecting_kwargs[param.name] = self.orig_args[i]
                elif param.name in inject:
                    # found in inject map
                    injecting_kwargs[param.name] = inject[param.name]
                elif param.annotation == logging.Logger:
                    # found an unbound logger
                    self.logger_param = param.name
                elif param.annotation == PipelineExecutionContext:
                    # found an unbound context
                    self.pipeline_param = param.name
                    injecting_kwargs[param.name] = None
                elif param.default == LastInput:
                    # found a 'LastInput'
                    self.last_input_param = param.name
                    injecting_kwargs[param.name] = None
                elif param.default is None and inject_fail_on_none:
                    inject_errors.append((param.name, None))

        if inject_errors:
            valerr = ''
            for err in inject_errors:
                if err[1] is None:
                    valerr += " {}\n".format(err[0])
                else:
                    valerr += " {} (using '{}')\n".format(err[0], err[1])
            raise RuntimeError(
                "xtz failed to inject parameters for '{}()':\n{}"
                .format(self.func.__name__, valerr))

        # insert the found injectables
        rebuilt_kwargs = copy.copy(self.orig_kwargs)
        for name, value in injecting_kwargs.items():
            rebuilt_kwargs[name] = value

        # rebind and call
        new_boundargs = funcsig.bind(*rebuilt_args, **rebuilt_kwargs)
        new_boundargs.apply_defaults()

        self.rebuilt_args = rebuilt_args
        self.rebuilt_kwargs = rebuilt_kwargs

    def execute(self, last_input=None, logger=None, context=None):
        if self.last_input_param is not None:
            self.rebuilt_kwargs[self.last_input_param] = last_input

        if self.pipeline_param is not None:
            self.rebuilt_kwargs[self.pipeline_param] = context

        if self.logger_param is not None:
            self.rebuilt_kwargs[self.logger_param] = logger

        returnobj = self.func(*self.rebuilt_args, **self.rebuilt_kwargs)
        return returnobj


def pipe(f=None, log_step=True):
    """Use the decorator @pipe on functions to define pipe steps"""

    def _pipe(f, *args, **kwargs):
        """Actual implementation"""
        if Pipeline.active_pipeline is None:
            # no active pipeline - just pass through
            return f(*args, **kwargs)
        else:
            if Pipeline.interactive_active:
                # in interactive repl
                defer_pipe_call = _DeferredPipeCall(
                    '', f, args, kwargs, log_step=log_step)
                Pipeline.active_pipeline.execute_interactive_step(
                    defer_pipe_call)
                return

            if Pipeline.active_context is not None:
                # active context (nested pipe calls)
                defer_pipe_call = _DeferredPipeCall(
                    '', f, args, kwargs, log_step=log_step)
                return Pipeline.active_pipeline.execute_runtime_step(defer_pipe_call)
            else:
                # record the step
                info = inspect.getframeinfo(inspect.stack()[2].frame)
                code_context = ""
                lineno = info.lineno

                # loop to actual function call to grab all lines
                while f.__name__ not in code_context and lineno > 0:
                    code_context = linecache.getline(info.filename,
                                                     lineno) + code_context
                    lineno -= 1
                code_context = textwrap.dedent(code_context.rstrip())

                defer_pipe_call = _DeferredPipeCall(
                    code_context, f, args, kwargs, log_step=log_step)
                Pipeline.active_pipeline.record_step(defer_pipe_call)
                return defer_pipe_call

    if f is not None:
        return decorator.decorate(f, _pipe)
    else:
        return decorator.decorator(_pipe)


class Pipeline(object):
    disable_colors: bool = False
    active_pipeline: Any = None
    active_context: Any = None
    interactive_active: bool = False
    interactive_last_function: str = None
    interactive_last_input: Any = None
    interactive_local_dict: Any = None
    interactive_cli: ptpython.repl.PythonCommandLineInterface = None

    def __init__(self,
                 inject: Mapping[str, object]=None,
                 logger: logging.Logger=None,
                 inject_fail_on_none=False) -> None:
        self.inject = inject
        self.logger = logger
        self.inject_fail_on_none = inject_fail_on_none
        self.steps: List[_DeferredPipeCall] = []

    def record(self) -> None:
        if Pipeline.active_pipeline is not None:
            raise RuntimeError("Can only one record one pipeline at a time")
        Pipeline.active_pipeline = self
        self.steps = []

    def record_step(self, defer_pipe_call: _DeferredPipeCall) -> None:
        if Pipeline.active_context is not None:
            raise RuntimeError("Can't record steps during pipeline execution")
        defer_pipe_call.bind(self.inject, self.inject_fail_on_none)
        self.steps.append(defer_pipe_call)

    def _print_color(self, color, message):
        if Pipeline.disable_colors:
            print(message)
        else:
            print('\x1b[' + color + 'm' + message + '\x1b[0m')

    def _gen_configure_repl(self):
        history_calls = {}

        # define a function that will be passed configure ptpython.repl
        def _configure_repl(repl: ptpython.repl.PythonRepl):
            repl.show_signature = True
            repl.show_docstring = True
            repl.insert_blank_line_after_output = False
            repl.true_color = True
            repl.history.append(
                "# this steps are copied directly from your program")
            # define the history
            for step in self.steps:
                history_calls[step.func.__name__] = step.func
                repl.history.append(step.orig_call_str + "\n")
            repl.history.append("")

        # return the tuple
        return (history_calls, _configure_repl)

    def _execute_interactive(self, frame, filename) -> None:
        Pipeline.interactive_local_dict = copy.copy(frame.f_locals)

        Pipeline.interactive_local_dict['xtz_inject'] = self.inject
        Pipeline.interactive_local_dict['xtz_logger'] = self.logger
        Pipeline.interactive_local_dict[
            'xtz_context'] = Pipeline.active_context
        Pipeline.interactive_local_dict['xtz_steps'] = tuple(self.steps)

        (history_calls, configure_repl) = self._gen_configure_repl()

        # add additional items for each key value for the history calls
        for key, value in history_calls.items():
            Pipeline.interactive_local_dict[key] = value

        self._print_color(
            '1;30;40', "Press F3 to quickly execute steps in your pipeline. " +
            "See 'last_output' to view last output from a step.\n")

        # embed repl
        self._create_repl(
            frame.f_globals,
            Pipeline.interactive_local_dict,
            configure=configure_repl,
            title=os.path.basename(filename))

    def _create_repl(self, global_vals, local_vals, configure=None,
                     title=None):
        """Create ptpython REPL by hand to support colors"""

        eventloop = ptpython.repl.create_eventloop()

        def get_globals():
            """return passed globals"""
            return global_vals

        def get_locals():
            """return passed locals"""
            return local_vals

        # Create REPL.
        repl = ptpython.repl.PythonRepl(get_globals, get_locals, vi_mode=True)

        if title:
            repl.terminal_title = title

        if configure:
            configure(repl)

        Pipeline.interactive_cli = ptpython.repl.PythonCommandLineInterface(
            python_input=repl, eventloop=eventloop)

        # Start repl.
        patch_context = Pipeline.interactive_cli.patch_stdout_context(
            raw=True)  # enable colors

        with patch_context:
            Pipeline.interactive_cli.run()

    def execute_interactive_step(self, step):
        step.bind(self.inject, self.inject_fail_on_none)
        if Pipeline.interactive_cli.in_paste_mode:
            self._print_color(
                '1;30;40',
                "Inspecting arguments to {}".format(step.func.__name__))
            self._print_color('1;30;40', "{}".format(step.inspect()))
            return

        if Pipeline.interactive_last_input is not None:
            self._print_color('1;30;40', "Using '{}' from '{}''".format(
                repr(Pipeline.interactive_last_input),
                Pipeline.interactive_last_function))
        Pipeline.interactive_last_function = step.func.__name__
        start = time.time()
        Pipeline.interactive_last_input = step.execute(
            Pipeline.interactive_last_input, Pipeline.active_context)
        Pipeline.interactive_local_dict[
            'last_output'] = Pipeline.interactive_last_input
        end = time.time()
        print('\x1b[3;32;40m' + "Finished {}() in {}".format(
            step.func.__name__, datetime.timedelta(seconds=end - start)) +
              '\x1b[0m')

    def execute_runtime_step(self, step):
        # simply inject and execute
        step.bind(self.inject, self.inject_fail_on_none)
        return step.execute(
            last_input=None,
            logger=self.logger.getChild(step.func.__name__),
            context=Pipeline.active_context)

    def execute(self, interactive=False) -> object:
        Pipeline.active_context = PipelineExecutionContext(
            datetime.datetime.utcnow(), self, interactive)

        if interactive:
            (frame, filename, *_) = inspect.stack()[1]
            Pipeline.interactive_active = True
            self._execute_interactive(frame, filename)
            Pipeline.interactive_active = False
            Pipeline.active_pipeline = None
            Pipeline.active_context = None
            return None

        total_steps = sum([step.log_step for step in self.steps])

        if self.logger != None:
            self.logger.info("Executing %d steps", total_steps)
            for i, step in enumerate(
                [step for step in self.steps if step.log_step]):
                self.logger.info("\t%d '%s'", i + 1, step.func.__name__)

            self.logger.info(">>> Beginning pipeline execution...")

        total_start = time.time()
        last_input = None

        i = 0
        for step in self.steps:
            if self.logger != None:
                start_group(step.func.__name__)

            if step.log_step:
                i = i + 1
                if self.logger != None:
                    self.logger.info(">> Starting step %d of %d", i,
                                     total_steps)

            start = time.time()
            try:
                last_input = step.execute(last_input, self.logger,
                                          Pipeline.active_context)
            except:
                if self.logger != None:
                    self.logger.exception(
                        "Exception while executing '%s()' at step %d",
                        step.func.__name__, i)
                Pipeline.active_pipeline = None
                raise
            end = time.time()

            if self.logger != None:
                end_group()
                if step.log_step:
                    self.logger.info(
                        ">> Finished step %d of %d in %s",
                        i,
                        total_steps,
                        datetime.timedelta(seconds=end - start))
        total_end = time.time()

        if self.logger != None:
            self.logger.info(
                ">>> Finished pipeline execution in %s",
                datetime.timedelta(seconds=total_end - total_start))

        Pipeline.active_pipeline = None
        Pipeline.active_context = None
        return last_input


class PipelineExecutionContext(object):
    def __init__(self,
                 start_time: datetime.datetime,
                 pipeline: Pipeline,
                 is_interactive: bool) -> None:
        self._start_time = start_time
        self._pipeline = pipeline
        self._is_interactive = is_interactive
        self._end_time: datetime.datetime = None
        self._log_stack: List[logging.Logger] = None

    @property
    def start_time(self):
        return self._start_time

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def is_interactive(self):
        return self._is_interactive

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value: datetime.datetime):
        self._end_time = value

    @property
    def log_stack(self):
        return self._log_stack

    @log_stack.setter
    def log_stack(self, value: List[logging.Logger]):
        self._log_stack = value


def run(steps: List[_DeferredPipeCall],
        inject: Mapping[str, object]=None,
        logger: logging.Logger=None,
        inject_fail_on_none=False) -> object:
    # TODO - make *steps
    pipeline: Pipeline = Pipeline(
        inject=inject, logger=logger, inject_fail_on_none=inject_fail_on_none)
    pipeline.record()
    for step in steps:
        pipeline.record_step(step)
    return pipeline.execute()


@pipe(log_step=False)
def start_group(group: str, context: PipelineExecutionContext=None):
    # store a stack on the context
    if context.log_stack is None:
        context.log_stack = [context.pipeline.logger]
    else:
        context.log_stack.append(context.pipeline.logger)

    # reassign logger
    context.pipeline.logger = context.pipeline.logger.getChild(group)


@pipe(log_step=False)
def end_group(context: PipelineExecutionContext=None):
    if context.log_stack:
        context.pipeline.logger = context.log_stack.pop()
