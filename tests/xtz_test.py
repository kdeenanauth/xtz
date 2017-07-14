#pylint: disable=locally-disabled, missing-docstring, protected-access, W0613, E1101

import logging
import unittest
import unittest.mock

import inspect
import linecache
import textwrap
import decorator
import xtz

@xtz.pipe
def step_one(one: int):
    return one

@xtz.pipe
def step_two(two: int):
    return two

@xtz.pipe
def with_injection(one: int, injecta: str = None, logger: logging.Logger=None):
    logger.info(' injecta:{}'.format(injecta))
    return injecta

@xtz.pipe
def as_nested(one: int, injectb: str = None, logger: logging.Logger=None):
    logger.info(' injectb:{}'.format(injectb))
    nested_val = with_injection(one)
    logger.info(" returnval:{}".format(nested_val))
    return nested_val

def test_deferred(func=None, log_step=True):
    """Use the decorator @test_deferred on functions to return a deferred pipe call"""

    def _test_deferred(func, *args, **kwargs):
        """Actual implementation"""
        # record the step
        info = inspect.getframeinfo(inspect.stack()[2].frame)
        code_context = ""
        lineno = info.lineno

        # loop to actual function call to grab all lines
        while func.__name__ not in code_context and lineno > 0:
            code_context = linecache.getline(info.filename,
                                             lineno) + code_context
            lineno -= 1
        code_context = textwrap.dedent(code_context.rstrip())
        defer_pipe_call = xtz.xtz._DeferredPipeCall(
            code_context, func, args, kwargs, log_step=log_step)
        return defer_pipe_call

    if func is not None:
        return decorator.decorate(func, _test_deferred)
    else:
        return decorator.decorator(_test_deferred)

@test_deferred
def testdeferred_stepone(one: int):
    return one

@test_deferred
def testdeferred_injection(one: int, injecta: str = None):
    return injecta

@test_deferred
def testdeferred_injection_with_logger(one: int, injecta: str=None, logger: logging.Logger=None):
    return logger

@test_deferred
def testdeferred_pipeline_context(context: xtz.PipelineExecutionContext=None):
    return context

class TestDeferredPipeCall(unittest.TestCase):
    def test_simple_bind(self):
        dpc = testdeferred_stepone(1)
        dpc.bind()
        self.assertEqual(dpc.execute(), 1)

    def test_kwarg_bind(self):
        dpc = testdeferred_injection(1, injecta='testa')
        dpc.bind()
        self.assertEqual(dpc.execute(), 'testa')

    def test_inject_override_bind(self):
        dpc = testdeferred_injection(1, injecta='testa')
        dpc.bind({'injecta' : 'testb'})
        self.assertEqual(dpc.execute(), 'testa')

    def test_inject_bind(self):
        dpc = testdeferred_injection(1)
        dpc.bind({'injecta' : 'testb'})
        self.assertEqual(dpc.execute(), 'testb')

    def test_inject_logger_none(self):
        dpc = testdeferred_injection_with_logger(1, 'testa')
        dpc.bind()
        self.assertEqual(dpc.execute(), None)

    def test_inject_logger(self):
        dpc = testdeferred_injection_with_logger(1)
        dpc.bind()
        log = logging.getLogger()
        self.assertEqual(dpc.execute(logger=log), log)

    def test_pipeline_context(self):
        dpc = testdeferred_pipeline_context()
        dpc.bind()
        context = xtz.PipelineExecutionContext(None, None, False)
        self.assertEqual(dpc.execute(context=context), context)


@unittest.skip
class TestPipeDecorator(unittest.TestCase):
    def test_execution_no_pipeline(self):
        no_pipe_result = step_one(1)
        self.assertEqual(no_pipe_result, 1)

    def test_execution_with_pipeline(self):
        pipeline = xtz.Pipeline()
        pipeline.record()
        defer_pipe = step_one(1)
        self.assertIsInstance(defer_pipe, xtz.xtz._DeferredPipeCall)
        self.assertEqual(defer_pipe.execute(), 1)
        pipeline.execute()

@unittest.skip
class TestPipelines(unittest.TestCase):

    def get_logger(self):
        return unittest.mock.create_autospec(logging.Logger)

    def test_retval(self):
        logger = self.get_logger()
        pipeline = xtz.Pipeline(
            inject={'injecta': 'valuea',
                    'injectb': 'valueb'}, logger=logger)
        pipeline.record()
        step_retval = step_one(1)
        self.assertIsInstance(step_retval, xtz.xtz._DeferredPipeCall)
        pipeline_retval = pipeline.execute()
        self.assertEqual(pipeline_retval, 1)
        logger.info.assert_called()
        logger.exception.assert_not_called()

    def test_config(self):
        logger = self.get_logger()
        pipeline = xtz.Pipeline(
            inject={'injecta': 'test_config:valuea',
                    'injectb': 'test_config:valueb'}, logger=logger)
        pipeline.record()
        step_retval = with_injection(1)
        self.assertIsInstance(step_retval, xtz.xtz._DeferredPipeCall)
        pipeline_retval = pipeline.execute()
        self.assertEqual(pipeline_retval, 'test_config:valuea')

    def test_nested(self):
        #logger = self.get_logger()
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        pipeline = xtz.Pipeline(
            inject={'injecta': 'test_nested:valuea',
                    'injectb': 'test_nested:valueb'}, logger=logger)
        pipeline.record()
        step_retval = as_nested(1)
        self.assertIsInstance(step_retval, xtz.xtz._DeferredPipeCall)
        pipeline_retval = pipeline.execute()
        self.assertEqual(pipeline_retval, 'test_nested:valuea')


if __name__ == '__main__':
    unittest.main()
