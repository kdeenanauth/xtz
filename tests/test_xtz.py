#pylint: disable=locally-disabled, missing-docstring, protected-access, W0613

import logging
import unittest
import unittest.mock

import xtz

@xtz.pipe
def step_one(one: int):
    print('inside step_one')
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

class TestXtz(unittest.TestCase):

    def get_logger(self):
        return unittest.mock.create_autospec(logging.Logger)

    def ztest_retval(self):
        logger = self.get_logger()
        pipeline = xtz.Pipeline(
            inject={'injecta': 'valuea',
                    'injectb': 'valueb'}, logger=logger)
        pipeline.record()
        step_retval = step_one(1)
        self.assertTrue(callable(step_retval))
        pipeline_retval = pipeline.execute()
        self.assertEqual(pipeline_retval, 1)
        logger.info.assert_called()
        logger.exception.assert_not_called()

    def ztest_config(self):
        logger = self.get_logger()
        pipeline = xtz.Pipeline(
            inject={'injecta': 'test_config:valuea',
                    'injectb': 'test_config:valueb'}, logger=logger)
        pipeline.record()
        step_retval = with_injection(1)
        self.assertTrue(callable(step_retval))
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
        self.assertTrue(callable(step_retval))
        pipeline_retval = pipeline.execute()
        self.assertEqual(pipeline_retval, 'test_nested:valuea')


if __name__ == '__main__':
    unittest.main()
