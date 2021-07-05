#### yacs

创建一个配置文件`config.py`

```python
from yacs.config import CfgNode as CN

__C = CN()

__C.SYSTEM.NUM_GPUS = 8
cfg = __C
```

然后使用

```python
from config.config import cfg#导入包
print(cfg.SYSTEM.NUM_GPUS)#获取配置信息
```

然后使用yaml文件更新

```python
cfg.merge_from_file(yaml_path)
```

将固定的config写入到`config.py`，不同组实验中的一些参数写入每个实验的yaml配置文件。

https://www.jianshu.com/p/f16631b69c65

#### colorama

https://www.cnblogs.com/xiao-apple36/p/9151883.html

#### logging 

This module defines functions and classes which implement a flexible event logging system for applications and libraries.

python的日志模块

日志级别严重程度从低到高：`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。具体使用时含义分别

| 级别       | 何时使用                                                     |
| ---------- | ------------------------------------------------------------ |
| `DEBUG`    | 细节信息，仅当诊断问题时使用                                 |
| `INFO`     | 确认程序按预期运行                                           |
| `WARNING`  | 表明有已经或即将发生的意外（例如：磁盘空间不足）。程序仍按预期进行 |
| `ERROR`    | 由于严重的问题，程序的某些功能已经不能正常执行               |
| `CRITICAL` | 严重的错误，表明程序已不能继续执行                           |

日志库采用模块化方法，并提供几类组件：记录器（Logger）、处理程序（Handler）、过滤器和格式化程序（Formatter）。

- 记录器暴露了应用程序代码直接使用的接口。

- 处理程序将日志记录（由记录器创建）发送到适当的目标。

- 过滤器提供了更精细的附加功能，用于确定要输出的日志记录。

- 格式化程序指定最终输出中日志记录的样式。

  ```python
  import logging
  
  # create logger
  logger = logging.getLogger('simple_example')
  logger.setLevel(logging.DEBUG)
  
  # create console handler and set level to debug
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  
  # create formatter
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  
  # add formatter to ch
  ch.setFormatter(formatter)
  
  # add ch to logger
  logger.addHandler(ch)
  
  # create file handler
  fh = logging.FileHandler('./log.txt')
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  
  # 'application' code
  logger.debug('debug message')
  logger.info('info message')
  logger.warning('warn message')
  logger.error('error message')
  logger.critical('critical message')
  ```

  同时，logging模块中的格式化方式可以看看LogRecord。

  另外，logging也支持配置初始化。

  **无配置文件**

  ```python
  import logging
  
  # create logger
  logger = logging.getLogger('simple_example')
  logger.setLevel(logging.DEBUG)
  
  # create console handler and set level to debug
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  
  # create formatter
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  
  # add formatter to ch
  ch.setFormatter(formatter)
  
  # add ch to logger
  logger.addHandler(ch)
  
  # 'application' code
  logger.debug('debug message')
  logger.info('info message')
  logger.warning('warn message')
  logger.error('error message')
  logger.critical('critical message')
  ```

  **使用配置文件**

  `.py`

  ```python
  import logging
  import logging.config
  
  logging.config.fileConfig('logging.conf')
  
  # create logger
  logger = logging.getLogger('simpleExample')
  
  # 'application' code
  logger.debug('debug message')
  logger.info('info message')
  logger.warning('warn message')
  logger.error('error message')
  logger.critical('critical message')
  ```

  `.conf`

  ```
  [loggers]
  keys=root,simpleExample
  
  [handlers]
  keys=consoleHandler
  
  [formatters]
  keys=simpleFormatter
  
  [logger_root]
  level=DEBUG
  handlers=consoleHandler
  
  [logger_simpleExample]
  level=DEBUG
  handlers=consoleHandler
  qualname=simpleExample
  propagate=0
  
  [handler_consoleHandler]
  class=StreamHandler
  level=DEBUG
  formatter=simpleFormatter
  args=(sys.stdout,)
  
  [formatter_simpleFormatter]
  format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
  datefmt=
  ```

  **yaml**版本

  ```yaml
  version: 1
  formatters:
    simple:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  handlers:
    console:
      class: logging.StreamHandler
      level: DEBUG
      formatter: simple
      stream: ext://sys.stdout
  loggers:
    simpleExample:
      level: DEBUG
      handlers: [console]
      propagate: no
  root:
    level: DEBUG
    handlers: [console]
  ```

  

  #### easydict

  ```python
  from easydict import EasyDict as edict
  ```

  相对于python原本的dict而言，edict能够使用下面的语法。也就是可以方便地应用 **.** 来访问dict的值。

  ```python
  d = edict()
  d.a = 'a'
  ```

  #### argparse

  类似Fire但是配置起来比较麻烦，但控制粒度高。

  #### json

  用于`json`操作的模块。