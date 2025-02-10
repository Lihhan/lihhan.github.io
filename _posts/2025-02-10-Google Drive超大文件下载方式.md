---
layout:     post
title:      Google Drive超大文件下载方式
subtitle:   解决方案日志
date:       2025-02-10
author:     lihan
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - blog
    - 配环境
---

因为跑实验需要从Google Drive上下几个几百G的大数据集，于是又要对付Google Drive了。

因为实验室的服务器可以科学上网，之前一直是linux wget（很慢且不稳定）或者直接win科学上网下载到本地再上传，但是众所周知谷歌网盘各种断线重连限速n件套简直不做人，之前也一直非常struggle；

这次正好来欧洲旅游，直接肉身翻墙尝试下载，然后确定了不是原来梯子有问题，而是真的google drive网页版比较垃圾。。


非常丢人，第一时间我想到的就是去闲鱼找一个外包代下载（该服务因为一些原因非常欣欣向荣），当对面看到我的IP在海外却要找代下的时候肯定非常无语但还是接下了单，本来代下载的市场价大概在0.5-1元/G，但我找到的小哥人非常好帮我封顶收了45元，于是便成交了。

接着小哥就开始和我攀谈，说我既然在海外为什么不自己下载；我说公寓网太烂网页一直断线重传，于是他就啼笑皆非地告诉了我可以使用rclone挂载谷歌网盘，接着直接下载，不会遇到断线或者限速的问题；

具体操作如下：

在win上，先在https://rclone.org/downloads/下载Intel/AMD - 64 Bit版本，接着cmd切换根目录之后rclone config，依次输入n-你想起的名字-20-(enter)-(enter)-1-再一路按enter下去就配好啦。然后rclone lsf xxx(你的用户名): --drive-shared-with-me就可以查看分享给你的网盘文件，譬如其中有一个hiss/文件夹想要下载到D盘，那么你接下来就可以运行rclone copy "xxx(你的用户名):hiss" D:\Downloads --drive-shared-with-me -P，就可以开始下载啦。
