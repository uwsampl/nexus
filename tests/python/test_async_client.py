import os
import random
import asyncio
from datetime import datetime, timedelta

import nexus

vgg_face_dir = '/home/abcdabcd987/datasets/vgg_face'

service_addr = "127.0.0.1:9001"

def load_images(root, maxlen):
    images = {}
    for fn in os.listdir(root)[:maxlen]:
        with open(os.path.join(root, fn), 'rb') as f:
            im = f.read()
            images[fn] = im
    return images


async def test_client(images, interval):
    images = iter(images)
    interval = timedelta(seconds=interval)
    user_id = random.randint(1, 1000000000)
    async with nexus.AsyncClient(service_addr, user_id) as client:
        pending = set()
        next_time = datetime.now()
        try:
            next_image = next(images)
        except StopIteration:
            return
        while True:
            timeout = (next_time - datetime.now()).total_seconds()
            if timeout > 0:
                await asyncio.sleep(timeout)
            else:
                while timeout <= 0 and next_image is not None:
                    next_time += interval
                    timeout = (next_time - datetime.now()).total_seconds()
                    pending.add(client.request(next_image))
                    try:
                        next_image = next(images)
                    except StopIteration:
                        next_image = None
                done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    print('==========', datetime.now(), task.result())
                if not pending and next_image is None:
                    break


if __name__ == "__main__":
    print('Test client...')
    images = list(load_images(vgg_face_dir, 20).values())

    print('Testing the non concurrent case')
    asyncio.run(test_client(images, 0.5))

    print('Testing the concurrent case')
    asyncio.run(test_client(images, 0.0001))
