import asyncio
import argparse
import random
import sys

import nexus


def read_image(img):
    if img == "-":
        return sys.stdin.buffer.read()
    with open(img, "rb") as f:
        return f.read()


async def query(server, image):
    user_id = random.randint(0, 2 ** 31 - 1)
    async with nexus.AsyncClient(server, user_id) as client:
        _send_time, _recv_time, reply = await client.request(image)
    print(reply)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file. `-` to read from stdin.")
    parser.add_argument("--server", help="Frontend server", default="localhost:9001")
    args = parser.parse_args()

    image = read_image(args.image)
    asyncio.run(query(args.server, image))


if __name__ == "__main__":
    main()
