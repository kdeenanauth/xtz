# xtz

The xtz module provides an environment to easily inject dependencies, log and debug
pipeline steps.

This module is still alpha quality. Feel free to use at your own risk.

## Example
Code:
```python
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
```

Executing:
```
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
```

Executing (with interactive):
```
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
```

## TODO:
* Log groupings better
* Event hooks
* Support injecting into class constructors
* Support async flows for non-linear pipelines
