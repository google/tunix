## Example 1: Write a process

Just define the `main` function, that's it.

```python
from tunix.experimental.distributed.runtime.context import ProcessContext

def main(argv, context: ProcessContext | None) -> None:
  print("hello world")
```

To run it

```shell
$ PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --process_main=basics.basic.main
```

Expect output

```shell
hello world
```

## Example 2: Add flag to process

Use `argv` argument.

```python
def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--message", type=str, default="", help="")
  args = parser.parse_args(argv)

  print(args.message)
```

To run it

```shell
$ PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --process_main=basics.flag.main --message="hello flag"
```

Expect output

```shell
hello flag
```

## Example 3: Discover other processes

This example shows how to write two processes and let them discover each other.

First, we start a `door` process and make it discoverable on port `12345` by setting `--discovery_id=door` and `--discovery_port=12345`.

Then, we start a `knocker` process and tell it to find `door` at address `door:12345` by setting `--discovery_addrs=door:12345`, and send a message by setting `--say="open the door"`.

To run it

1. Start door

    ```shell
    $ PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --process_main=basics.door.main --discovery_id=door --discovery_port=12345
    ```

2. Start knocker

    ```shell
    $ PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --process_main=basics.knocker.main --discovery_addrs=door:12345 --say="open the door"
    ```

3. Expected output

    ```shell
    # door.py
    this is door!
    discovery server started on port 12345
    localhost knocked and said: open the door
    discovery server stopped

    # knocker.py
    this is knocker!
    registered to discovery server at localhost:12345
    ```

## Example 4: Simulate RL workload

This example shows how to write a fake RL workload with 4 processes.
- one `orchestrator` process which drives the loop.
- two `rollout` processes which generates completion from prompt.
- one `trainer` process which generates weights from prompt and completion.

The RL workflow tries to learn the expected value of the addition of two random numbers range from 0 to 10.
- the prompt is a math expression, like `2 + 3`.
- the completion is the result, like `= 5`.
- errors are intentionally introduced to the completion randomly.
- the weights are updated 1% if the calculated result is correct.
- the weights are still updated 0.01% if the calculated result is wrong.
- after sufficient amount of steps, the weights should approximate 10.

Note, this example just simulates the data flow, don't try to relate it to actual RL algorithms.

To run it

1. Start orchestrator process

    ```shell
    $ PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_id=orchestrator --discovery_port=12345 --process_main=rl.orchestrator.main --max_train_step=1000
    ```

2. Start two rollout processes

    ```shell
    $ PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_addrs=orchestrator:12345 --process_main=rl.rollout.main --server_id=rollout-0 --server_port=11111
    $ PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_addrs=orchestrator:12345 --process_main=rl.rollout.main --server_id=rollout-1 --server_port=22222
    ```

3. Start trainer process

    ```shell
    $ PYTHONPATH=./examples/distributed python -m tunix.experimental.distributed.runtime.main --discovery_addrs=orchestrator:12345 --process_main=rl.trainer.main --server_id=trainer --server_port=33333
    ```
