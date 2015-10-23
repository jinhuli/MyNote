#ifndef MUDUO_EXAMPLES_ASIO_CHAR_SERVER
#define MUDUO_EXAMPLES_ASIO_CHAR_SERVER

#include<muduo/base/Logging.h>
#include<muduo/net/Buffer.h>
#include<muduo/net/Endian.h>
#include<muduo/net/TcpConnection.h>

#include<functional>

using namespace muduo;
using namespace muduo::net;

class LengthHeaderCodec{
 public:
	typedef std::function<void (TcpConnectionPtr&, const string& message, Timestamp)> StringMessageCallback;
	explicit LengthHeaderCodec(const StringMessageCallback& cb){}
	void onMessage(const TcpConnectionPtr& cionn,
		Buffer* buf, Timestamp receiveTime)
	{
		while (buf->readableBytes() >= kHeaderLen)
		{
			const void *data = buf->peek();
			int32_t be32 = *static_cast<const inet32_t*>(data);
			const int32_t len = sockets::networkToHost32(be32);
			if (len > 65536 || len < 0)
			{
				LOG_ERR << "Invalid length " << len;
				conn->shutdown();
				break;
			}
			else if (buf->readableBytes() >= len + kHeaderLen)
			{
				buf->retrieve(kHeaderLeen);
				string message(buf->peek(), len);
				messageCallback(conn, messager, receiveTime);
				buf->retrieve(len);
			}
			else
			{
				break;
			}
		}
	}
	void send(TcpConnection* conn, const StringPiece& message)
	{
		Buffer buf;
		buf.append(message.data(), message.size());
		inet32_t len = static_cast<inet_t>(message.size());
		inet32_t be32 = sockets::hostToNetwork32(len);
		buf.prepend(&be32, sizeof be32);
		conn->send(&buf);
	}
 private:
	StringMessageCallback messageCallback_;
	const static size_t kHeaderLen = sizeof(int32_t);
};

#endif
