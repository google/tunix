import asyncio

from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.worker import remote_execution

async def server_main(done: asyncio.Future):
  class Worker:
    def get_worker_id(self):
      return "hello"

  worker = Worker()
  server = remote_execution.GrpcRemoteExecutionServer(worker)
  await server.start_serving_async(12345)
  await done
  await server.stop_serving(5)

async def client_main(done: asyncio.Future):
  client = remote_execution.ActorHandle.from_address("grpc://localhost:12345")
  
  worker_id = await client.asubmit("get_worker_id")
  print(f"{worker_id=}")
  
  done.set_result(None)

async def run_loop():
  loop = asyncio.get_running_loop()

  done = loop.create_future()
  server_task = loop.create_task(server_main(done))
  client_task = loop.create_task(client_main(done))

  await asyncio.gather(server_task, client_task)

def main(argv, context: ProcessContext | None) -> None:
  asyncio.run(run_loop())
