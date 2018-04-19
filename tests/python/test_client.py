import os
import random
import nexus

_test_data = os.path.abspath(os.path.join(__file__, '../../data'))

service_addr = "127.0.0.1:9001"

def load_images(root):
    images = {}
    for fn in os.listdir(root):
        with open(os.path.join(root, fn), 'rb') as f:
            im = f.read()
            images[fn] = im
    return images


def test_client():
    user_id = random.randint(1, 1000000000)
    client = nexus.Client(service_addr, user_id)
    images = load_images(os.path.join(_test_data, 'images'))
    for fn in images:
        reply = client.request(images[fn])
        print(fn)
        print(reply)


if __name__ == "__main__":
    print("Test client...")
    test_client()
