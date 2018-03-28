import struct
import socket

from .proto import nnquery_pb2 as npb

MAGIC_NUMBER = 0xDEADBEEF
HEADER_SIZE = 12
# Message type
MSG_USER_REGISTER = 1
MSG_USER_REQUEST = 2
MSG_USER_REPLY = 3


class Client:
    def __init__(self, server_addr, user_id):
        self.server_addr = server_addr
        self.user_id = user_id
        self.req_id = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(1) # timeout after 1 second
        host, port = server_addr.split(':')
        try:
            self.sock.connect((host, int(port)))
        except:
            raise RuntimeError("Error in connecting to %s" % server_addr)
        self.register()


    def __del__(self):
        self.sock.close()


    def register(self):
        req = npb.RequestProto(user_id=self.user_id)
        msg = self._prepare_message(MSG_USER_REGISTER, req)
        self.sock.sendall(msg)
        reply = self._recv_reply()
        assert reply.status == 0
        

    def request(self, img):
        req = self._prepare_req(img)
        msg = self._prepare_message(MSG_USER_REQUEST, req)
        failed = 0
        while True:
            try:
                self.sock.sendall(msg)
                reply = self._recv_reply()
                break
            except socket.timeout:
                failed += 1
                if failed == 3:
                    return None
        return reply


    def _prepare_req(self, img):
        req = npb.RequestProto()
        req.user_id = self.user_id
        req.req_id = self.req_id
        req.input.data_type = npb.DT_IMAGE
        req.input.image.data = img
        req.input.image.format = npb.ImageProto.JPEG
        req.input.image.color = True
        self.req_id += 1
        return req


    def _prepare_message(self, msg_type, request):
        body = request.SerializeToString()
        header = struct.pack('!LLL', MAGIC_NUMBER, msg_type, len(body))
        return header + body


    def _recv_reply(self):
        body_length = self._recv_header()
        buf = self._read_nbytes(body_length)
        reply = npb.ReplyProto()
        reply.ParseFromString(buf)
        return reply


    def _recv_header(self):
        buf = self._read_nbytes(HEADER_SIZE)
        magic_no, msg_type, length = struct.unpack('!LLL', buf)
        assert magic_no == MAGIC_NUMBER
        assert msg_type == MSG_USER_REPLY
        return length

    
    def _read_nbytes(self, n):
        """ Read exactly n bytes from the socket.
            Raise RuntimeError if the connection closed before
            n bytes were read.
        """
        buf = ''
        while n > 0:
            data = self.sock.recv(n)
            if data == '':
                raise RuntimeError("Unexpected connection close")
            buf += data
            n -= len(data)
        return buf
