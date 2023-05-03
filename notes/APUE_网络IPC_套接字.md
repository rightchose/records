**套接字**是通信端口的抽象。应用程序使用**套接字描述符**访问套接字。套接字描述符在UNIX系统中被当作是一种文件描述符。

#### 创建套接字

```c
#include <sys/socket.h>
int socket(int domain, int type, int protocol);
```

domain（域）确定通信的特性，例如地址格式。有以下几个地址族可以选择。

- AF_INET：IPv4因特网域
- AF_INET6：IPv6因特网域
- AF_UNIX：UNIX域
- AF_UNSPEC：未指定

type确定套接字类型，进一步确定通信特征。

- SOCKET_DGRAM：固定长度的、无连接的、不可靠的报文传递
- SOCKET_RAW：IP协议的数据报接口
- SOCKET_SEQPACKET：固定长度的、有序的、可靠的、面向连接的报文传递
- SOCK_STREAM：有序的、可靠的、双向的、面向连接的字节流

对于数据报（SOCKET_DGRAM)而言，两个对等进程之间通信不需要逻辑接口，只需要向对等进程所使用的套接字送出一个报文。因此提供了无连接的服务。

对于字节流（SOCKET_STREAM)要求在交换数据前，在本地套接字和通信对等进程的套接字之间建立一个逻辑连接。

protocol通常为0，表示给定域的套接字类型选择默认协议。

```c
#include <sys/socket.h>
int shutdown(int sockfd, int how);
```

禁止一个套接字的I/O。

**寻址**

目标进程标识：计算机网络地址，端口号 。

处理器字节序和网络字节序转换

```c
#include <arpa/inet.h>

uint32_t htonl(uint32_t hostint32); // 返回网络字节序表示的32位整数

uint16_t htons(uint16_t hostint16); // 返回网路字节序表示的16位整数

uint32_t ntohl(uint32_t netint32); //以主机字节序表示的32位整数

uint15_t ntohs(uint16_t netint16); // 以主机字节序表示的16位整数
```

**地址格式**

地址标识一个特定通信域的套接字端点。因此地址格式和通信域相关。为了使得不同格式的地址能够传入到套接字函数，地址会被强制转换成一个通用的地址结构sockaddr。

```
struct socketaddr {
	sa_family_t sa_family;	// 地址族
	char		sa_data[];	// 变长地址
};
```

socketaddr_int、socketaddr_in16分别标识IPv4、IPv6。但都会被强制转换成socketaddr。

BSD网络软件报包含inet_addr和inet_ntoa用于二进制地址格式和点分十进制字符表示之间的转换。但只适用于IPv4。有两个新函数inet_ntop和inet_pton既可以用于IPv4，也可以用于IPv6。

```c
#include <arpa/inet.h>

const char *inet_ntop(int domain, const void *restrict addr, char *restrict str, socklen_t size);

int inet_pton(int domain, const char *restrict str, void *restrict addr);
```

**将套接字和地址关联**

```c
#include <sys/socket.h>

int bind(int sockfd, const struct sockaddr *addr, socklen_t len);
```

如果要查看绑定到套接字上的地址

```c
#include <sys/socket.h>

int getsockname(int sockfd, struct sockaddr *restrict addr, socklen_t *restrict alenp);
```

如果套接字已经和对等方连接，可以调用getpeername函数来找到对方的地址

```c
#include <sys/socket.h>

int getpeername(int sockfd, struct sockaddr *restrict addr, socklen_t *restrict alenp);
```

**建立连接**

如果要处理一个面向连接的网络服务，也就是先前的SOCK_STREAM或SOCK_SEQPACKET，面向连接。那么在开始交换数据前，需要在请求服务的进程套接字（客户端）和提供服务的进程套接字（服务器）之间建立一个连接。使用connect函数来建立连接。

```c
#include <sys/socket.h>

int connect(int sockfd, const struct sockaddr *addr, socklen_t len);
```

**服务器监听**

```c
#include <sys/socket.h>

int listen(int sockfd, int backlog);
```

一旦服务器调用listen，所用的套接字就能接受连接客户端请求。服务器再使用accept函数获得连接请求并建立了一条连接。

```c
#include <sys/socket.h>

int accept(int sockfd, struct sockaddr *restrict addr, socklen_t *restrict len);
```

函数accept返回一个文件描述符，本质是个套接字描述符，该描述符连接到调用connect的客户端。该套接字描述符和原始套接字（sockfd）具有相同的套接字类型和地址族，但原始套接字并没有和建立的连接关联，而是继续保持可用状态，并接收其他连接请求。

另外，如果服务器调用accept，并且当前没有连接请求，服务器会阻塞直到一个请求到来。

**数据传输**

```c
#include <sys/socket.h>

ssize_t send(int sockfd, const void* buf, size_t nbytes, int flags);
```

类似write，使用send时，套接字必须已经连接。参数buf和nbytes的含义与write中的一致。相较于write，send支持第四个参数flags。

但send即使成功返回并不代表接收方一定收到了数据，send成功只保证数据已经被无错误地发送到网络驱动程序上。

除此之外还有sento。其和send的区别在于，sendto可以在无连接的套接字上指定一个目标地址。

```c
#include <sys/socket.h>

ssize_t sendto(int sockfd, const void *buf, size_t nbytes, int flags, const struct sockaddr *destaddr, socklen_t destlen);
```

对于有连接的套接字，目标地址可以忽略，因为隐含了目标地址。无连接的套接字除非先前调用connect，否则不能使用send。sendto提供了一种支持。

使用msghdr发送数据

```c
#include <sys/socket.h>

ssize_t sendmsg(int sockfd, const struct msghdr, int flags);
```

可以指定多重缓冲区来传输数据。

对于接受数据可以使用recv

```c
#include <sys/socket.h>

ssize_t recv(int sockfd, void *buf, size_t nbytes, int flags);
```

对于flags参数，有

- MSG_CMSG_CLOEXEC
- MSG_DONTWAIT：启动非阻塞（相当于O_NOBLOCK）
- MSG_ERRQUEUE：接收错误信息作为辅助数据
- MSG_OOB：如果协议支持，获取带外数据
- MSG_PEEK：返回数据包内容而不真正取走数据包
- MSG_TRUNC：即使数据包被截断，也返回数据包实际长度
- MSG_WAITALL：直到所有数据可用（仅限于SOCK_STREAM）

如果想获取数据发送者的源地址可以使用recvfrom。

```c
#include <sys/socket.h>

ssize_t recvfrom(int sockfd, void *restrict buf, size_t len, int flags, 
	strut sockaddr *restrict addr,
	socklen_t *restrict addrlen
);
```

使用readmsg接受数据，为了将接收到的数据送入多个缓存区

```
#include <sys/socket.h>


```

