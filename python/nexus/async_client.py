import struct
import socket
import asyncio
from datetime import datetime

from .proto import nnquery_pb2 as npb

MAGIC_NUMBER = 0xDEADBEEF
HEADER_SIZE = 12
# Message type
MSG_USER_REGISTER = 1
MSG_USER_REQUEST = 2
MSG_USER_REPLY = 3


class AsyncClient:
    def __init__(self, server_addr, user_id):
        self._server_addr = server_addr
        self._user_id = user_id
        self._req_id = 0
        self._reader_lock = asyncio.Lock()
        self._replies = {}

    @property
    def next_req_id(self):
        return self._req_id

    async def __aenter__(self):
        host, port = self._server_addr.split(':')
        self._reader, self._writer = await asyncio.open_connection(host, port)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._writer.close()
        await self._writer.wait_closed()

    async def register(self):
        req = npb.RequestProto(user_id=self.user_id)
        msg = self._prepare_message(MSG_USER_REGISTER, req)

        self._writer.write(msg)
        await self._writer.drain()

        reply, time = await self._wait_reply(req.req_id)
        assert reply.status == 0

    async def _do_request(self, req, msg):
        send_time = datetime.now()
        self._writer.write(msg)
        await self._writer.drain()

        reply, recv_time = await self._wait_reply(req.req_id)
        return send_time, recv_time, reply

    def request(self, img):
        req = self._prepare_req(img)
        msg = self._prepare_message(MSG_USER_REQUEST, req)
        return self._do_request(req, msg)

    def _prepare_req(self, img):
        req = npb.RequestProto()
        req.user_id = self._user_id
        req.req_id = self._req_id
        req.input.data_type = npb.DT_IMAGE
        req.input.image.data = img
        req.input.image.format = npb.ImageProto.JPEG
        req.input.image.color = True
        self._req_id += 1
        return req

    def request_with_hack_filename(self, filename):
        req = npb.RequestProto()
        req.user_id = self._user_id
        req.req_id = self._req_id
        req.input.data_type = npb.DT_IMAGE
        req.input.image.hack_filename = filename
        req.input.image.format = npb.ImageProto.JPEG
        req.input.image.color = True
        self._req_id += 1

        msg = self._prepare_message(MSG_USER_REQUEST, req)
        return self._do_request(req, msg)

    def _prepare_message(self, msg_type, request):
        body = request.SerializeToString()
        header = struct.pack('!LLL', MAGIC_NUMBER, msg_type, len(body))
        return header + body

    async def _wait_reply(self, req_id):
        while True:
            async with self._reader_lock:
                reply = self._replies.pop(req_id, None)
                if reply is not None:
                    return reply

                buf = await self._reader.readexactly(HEADER_SIZE)
                magic_no, msg_type, body_length = struct.unpack('!LLL', buf)
                assert magic_no == MAGIC_NUMBER
                assert msg_type == MSG_USER_REPLY

                buf = await self._reader.readexactly(body_length)
                reply = npb.ReplyProto()
                reply.ParseFromString(buf)
                self._replies[reply.req_id] = (reply, datetime.now())

                # return early to avoid lock competition
                reply = self._replies.pop(req_id, None)
                if reply is not None:
                    return reply
