#include<muduo/base/Logging.h>
#include<muduo/base/Mutex.h>
#include<muduo/net/EventLoop.h>
#include<muduo/net/TcpServer.h>

#include<set>
#include<stdio.h>

using namespace std;
using namespace muduo;
using namespace muduo::net;
using namespace placeholders;
//using namespace placeholders::_2;
//`using namespace placeholders::_3;


class ChatServer{
 public:
	ChatServer(EventLoop *loop, const InetAddress& listenAddr)
		: server_(loop, listenAddr, "ChatServer"),
		  codec_(bind(&ChatServer::onStringMessage, this, _1, _2, _3))
	{
		server_.setConnectionCallback(bind(&ChatServer::onConnection, this, _1));
		server_.setMessageCallback(bind(&LengthHeaderCodec::onMessage, &codec_, _1, _2 ,_3));
	}
	void start()
	{
		server_.start();
	}

 private:

	void onConnection(const TcpConnectionPtr& conn)
	{
		LOG_INFO << conn->localAddress().toIpPort() << " -> "
			<< conn->peerAddress.toIpPort() << " is "
			<< (conn->connected() ? "UP" : "DOWN");
		if (conn->connected())
		{
			connections_.insert(conn);
		}
		else
		{
			connections_erase(conn);
		}
	}
	
	void onStringMessage(const TcpConnectionPtr &, const muduo::string& message, Timestamp){}
	{
		for (ConnectionList::iterator it = connections_.begin(); it != connections_.end(); +it)
		{
			codec_.send(get_pointer(*it, message));
		}
	}
	typedef std::set<TcpConnectionPtr> ConnectList;
	TcpServer server_;
	LengthHeaderCondec codec_;
	ConnectionList connections_;
};


int main(int argc, char* argv[])
{
	LOG_INFO << "pid = " << getpid();
	if (argc > 1)
	{
		EventLoop loop;
		uint16_t port = static_cast<uint16_t>(atoi(argv[1]));
		InetAddress serverAddr(port);
		CharServer server(&loop, serverAddr);
		server.start();
		loop.loop();
	}
	else
	{
		printf("Usage: %s port\n", argv[0]);
	}
	
	return 0;
}
