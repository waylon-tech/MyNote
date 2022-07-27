## 目录

[toc]

## 查看本地`ip`

```shell
ubuntu@VM-16-12-ubuntu:~$ ifconfig
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.16.16.12  netmask 255.255.240.0  broadcast 172.16.31.255
        inet6 fe80::5054:ff:fe39:7b2c  prefixlen 64  scopeid 0x20<link>
        ether 52:54:00:39:7b:2c  txqueuelen 1000  (Ethernet)
        RX packets 282036554  bytes 58750182471 (58.7 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 329610220  bytes 48330289145 (48.3 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 12861  bytes 36890405 (36.8 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 12861  bytes 36890405 (36.8 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

```shell
user@tan-desktop:~$ netstat -aptn
(Not all processes could be identified, non-owned process info
 will not be shown, you would have to be root to see it all.)
Active Internet connections (servers and established)
Proto Recv-Q Send-Q Local Address           Foreign Address         State       PID/Program name
tcp        0      0 127.0.0.1:6010          0.0.0.0:*               LISTEN      -               
tcp        0      0 127.0.0.1:6011          0.0.0.0:*               LISTEN      -               
tcp        0      0 127.0.0.1:3476          0.0.0.0:*               LISTEN      -               
tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN      -               
tcp        0      0 127.0.0.1:631           0.0.0.0:*               LISTEN      -               
tcp        0    352 172.20.72.13:22         172.24.65.9:1164        ESTABLISHED -               
tcp        0      0 172.20.72.13:22         172.24.65.9:2059        ESTABLISHED -               
tcp6       0      0 ::1:6010                :::*                    LISTEN      -               
tcp6       0      0 ::1:6011                :::*                    LISTEN      -               
tcp6       0      0 :::22                   :::*                    LISTEN      -               
tcp6       0      0 ::1:631                 :::*                    LISTEN      -  
```

## 查看外网`ip`

### `curl` 纯文本格式输出

```
curl icanhazip.com
curl ifconfig.me
curl curlmyip.com
curl ip.appspot.com
curl ipinfo.io/ip
curl ipecho.net/plain
curl www.trackip.net/i
```

### `curl` `JSON` 格式输出

```
curl ipinfo.io/json
curl ifconfig.me/all.json
curl www.trackip.net/ip?json (有点丑陋)
```

### `curl` `XML` 格式输出

```
curl ifconfig.me/all.xml
```

### `curl` 得到所有 `IP` 细节 （挖掘机）

```
curl ifconfig.me/all
```

### 使用 `DYDNS` （当你使用 `DYDNS` 服务时有用）

```
curl -s 'http://checkip.dyndns.org' | sed 's/.*Current IP Address: \([0-9\.]*\).*/\1/g'
curl -s http://checkip.dyndns.org/ | grep -o "[[:digit:].]\+"
```

### 使用 `wget` 代替 `curl`

```
wget http://ipecho.net/plain -O - -q ; echo
wget http://observebox.com/ip -O - -q ; echo
```

### 使用 `host` 和 `dig` 命令

如果有的话，你也可以直接使用 host 和 dig 命令。

```
host -t a dartsclink.com | sed 's/.*has address //'
dig +short myip.opendns.com @resolver1.opendns.com
```

### bash 脚本示例:

```
#!/bin/bash
PUBLIC_IP=`wget http://ipecho.net/plain -O - -q ; echo`
echo $PUBLIC_IP
```